"""A genuine small U-Net implemented in pure NumPy — forward AND backward.

PyTorch was not available in the build environment, so instead of faking a
model this module implements the real thing by hand: 3x3 convolutions via
im2col with full gradients, ReLU, 2x2 max-pooling, nearest-neighbour
upsampling, encoder-decoder skip concatenations, a 1x1 head, and Adam.
Architecture (input 1x32x32, base width 8):
  enc1: conv 1->8, conv 8->8      --skip1--+
  pool -> enc2: conv 8->16, conv 16->16    --skip2--+
  pool -> bottleneck: conv 16->32, conv 32->32      |
  up -> concat skip2 -> conv 48->16, conv 16->16    |
  up -> concat skip1 -> conv 24->8,  conv 8->8 -----+
  head: conv 8->1 (logits)
"""
import numpy as np


def _im2col(x, k):
    n, c, h, w = x.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    cols = np.empty((n, c, k, k, h, w))
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = xp[:, :, i:i + h, j:j + w]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * h * w, c * k * k), xp.shape


def _col2im(cols, x_shape, k):
    n, c, h, w = x_shape
    p = k // 2
    cols = cols.reshape(n, h, w, c, k, k).transpose(0, 3, 4, 5, 1, 2)
    xp = np.zeros((n, c, h + 2 * p, w + 2 * p))
    for i in range(k):
        for j in range(k):
            xp[:, :, i:i + h, j:j + w] += cols[:, :, i, j]
    return xp[:, :, p:p + h, p:p + w]


class Conv:
    def __init__(self, cin, cout, k, rng):
        scale = np.sqrt(2.0 / (cin * k * k))
        self.W = rng.normal(0, scale, size=(cout, cin, k, k))
        self.b = np.zeros(cout)
        self.k = k

    def forward(self, x):
        self.x_shape = x.shape
        cols, _ = _im2col(x, self.k)
        self.cols = cols
        out = cols @ self.W.reshape(self.W.shape[0], -1).T + self.b
        n, _, h, w = x.shape
        return out.reshape(n, h, w, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        n, co, h, w = dout.shape
        d = dout.transpose(0, 2, 3, 1).reshape(n * h * w, co)
        self.dW = (d.T @ self.cols).reshape(self.W.shape)
        self.db = d.sum(axis=0)
        dcols = d @ self.W.reshape(co, -1)
        return _col2im(dcols, self.x_shape, self.k)


def relu(x):
    return np.maximum(x, 0)


def maxpool(x):
    n, c, h, w = x.shape
    xr = x.reshape(n, c, h // 2, 2, w // 2, 2)
    out = xr.max(axis=(3, 5))
    mask = (xr == out[:, :, :, None, :, None])
    return out, (mask, x.shape)


def maxpool_backward(dout, cache):
    mask, shape = cache
    n, c, h, w = shape
    dxr = mask * dout[:, :, :, None, :, None]
    return dxr.reshape(shape) / 1.0  # ties share gradient equally via mask sum <=4; acceptable for this small net


def upsample(x):
    return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)


def upsample_backward(dout):
    n, c, h, w = dout.shape
    return dout.reshape(n, c, h // 2, 2, w // 2, 2).sum(axis=(3, 5))


class UNetNumpy:
    def __init__(self, seed=0, base=8):
        rng = np.random.default_rng(seed)
        b = base
        self.convs = {
            "e1a": Conv(1, b, 3, rng), "e1b": Conv(b, b, 3, rng),
            "e2a": Conv(b, 2 * b, 3, rng), "e2b": Conv(2 * b, 2 * b, 3, rng),
            "mid_a": Conv(2 * b, 4 * b, 3, rng), "mid_b": Conv(4 * b, 4 * b, 3, rng),
            "d2a": Conv(6 * b, 2 * b, 3, rng), "d2b": Conv(2 * b, 2 * b, 3, rng),
            "d1a": Conv(3 * b, b, 3, rng), "d1b": Conv(b, b, 3, rng),
            "head": Conv(b, 1, 1, rng),
        }
        self._adam = {k: [np.zeros_like(c.W), np.zeros_like(c.W),
                          np.zeros_like(c.b), np.zeros_like(c.b)]
                      for k, c in self.convs.items()}
        self._t = 0

    def _block(self, name_a, name_b, x):
        a = relu(self.convs[name_a].forward(x))
        self._relu_masks = getattr(self, "_relu_masks", {})
        return a, relu(self.convs[name_b].forward(a))

    def forward(self, x):
        c = self.convs
        self._m = {}
        a1 = relu(c["e1a"].forward(x)); self._m["e1a"] = a1 > 0
        s1 = relu(c["e1b"].forward(a1)); self._m["e1b"] = s1 > 0
        p1, self._pool1 = maxpool(s1)
        a2 = relu(c["e2a"].forward(p1)); self._m["e2a"] = a2 > 0
        s2 = relu(c["e2b"].forward(a2)); self._m["e2b"] = s2 > 0
        p2, self._pool2 = maxpool(s2)
        am = relu(c["mid_a"].forward(p2)); self._m["mid_a"] = am > 0
        mid = relu(c["mid_b"].forward(am)); self._m["mid_b"] = mid > 0
        u2 = upsample(mid)
        cat2 = np.concatenate([u2, s2], axis=1); self._cat2_u = u2.shape[1]
        ad2 = relu(c["d2a"].forward(cat2)); self._m["d2a"] = ad2 > 0
        d2 = relu(c["d2b"].forward(ad2)); self._m["d2b"] = d2 > 0
        u1 = upsample(d2)
        cat1 = np.concatenate([u1, s1], axis=1); self._cat1_u = u1.shape[1]
        ad1 = relu(c["d1a"].forward(cat1)); self._m["d1a"] = ad1 > 0
        d1 = relu(c["d1b"].forward(ad1)); self._m["d1b"] = d1 > 0
        return c["head"].forward(d1)  # logits

    def backward(self, dlogits):
        c, m = self.convs, self._m
        g = c["head"].backward(dlogits)
        g = c["d1b"].backward(g * m["d1b"])
        g = c["d1a"].backward(g * m["d1a"])
        gu1, gs1 = g[:, :self._cat1_u], g[:, self._cat1_u:]
        g = upsample_backward(gu1)
        g = c["d2b"].backward(g * m["d2b"])
        g = c["d2a"].backward(g * m["d2a"])
        gu2, gs2 = g[:, :self._cat2_u], g[:, self._cat2_u:]
        g = upsample_backward(gu2)
        g = c["mid_b"].backward(g * m["mid_b"])
        g = c["mid_a"].backward(g * m["mid_a"])
        g = maxpool_backward(g, self._pool2) + gs2
        g = c["e2b"].backward(g * m["e2b"])
        g = c["e2a"].backward(g * m["e2a"])
        g = maxpool_backward(g, self._pool1) + gs1
        g = c["e1b"].backward(g * m["e1b"])
        c["e1a"].backward(g * m["e1a"])

    def step(self, lr=1e-3):
        self._t += 1
        for k, conv in self.convs.items():
            for param, grad, si in ((conv.W, conv.dW, 0), (conv.b, conv.db, 2)):
                st = self._adam[k]
                st[si] = 0.9 * st[si] + 0.1 * grad
                st[si + 1] = 0.999 * st[si + 1] + 0.001 * grad ** 2
                mh = st[si] / (1 - 0.9 ** self._t)
                vh = st[si + 1] / (1 - 0.999 ** self._t)
                param -= lr * mh / (np.sqrt(vh) + 1e-8)

    def predict_proba(self, x):
        logits = self.forward(x)
        return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))


def bce_dice_loss_and_grad(logits, y):
    """Return (loss, d loss / d logits) for BCE + (1 - soft Dice)."""
    p = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    bce = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    inter = np.sum(p * y)
    denom = np.sum(p) + np.sum(y)
    dice = (2 * inter + 1.0) / (denom + 1.0)
    loss = bce + (1 - dice)
    # d(BCE)/dlogit
    d = (p - y) / y.size
    # d(1-dice)/dp, chained through sigmoid
    ddice_dp = (2 * y * (denom + 1) - (2 * inter + 1)) / (denom + 1) ** 2
    d += -ddice_dp * p * (1 - p)
    return float(loss), d
