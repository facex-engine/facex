"""Reproduce numerical examples in docs/research/54_math_solutions_ru.md."""
from itertools import combinations, permutations, product
import math

import numpy as np


def verify():
    checked = []
    rng = np.random.default_rng(54)

    def check(number, condition):
        if not condition:
            raise AssertionError(f"Example {number}")
        checked.append(number)

    def fit(z, y, w):
        zm, ym = np.average(z, weights=w), np.average(y, weights=w)
        c = np.sum(w*(z-zm).conj()*(y-ym)) / np.sum(w*abs(z-zm)**2)
        return c, ym - c*zm

    z = rng.normal(size=5) + 1j*rng.normal(size=5)
    w = np.arange(1,6)
    y = (.8+.4j)*z + (4-3j)
    c,t = fit(z,y,w)
    check(1, np.allclose([c,t],[.8+.4j,4-3j]))
    e = .01*(rng.normal(size=5) + 1j*rng.normal(size=5))
    spread = sum(w*abs(z-np.average(z,weights=w))**2)
    check(2, abs(fit(z,y+e,w)[0]-c) <= math.sqrt(sum(w*abs(e)**2)/spread))
    W,Z,Y,U,V = sum(w),sum(w*z),sum(w*(y+e)),sum(w*abs(z)**2),sum(w*z.conj()*(y+e))
    for i in range(5):
        keep = np.arange(5) != i
        n = W-w[i]
        a,b = Z-w[i]*z[i],Y-w[i]*(y[i]+e[i])
        reduced = (V-w[i]*z[i].conj()*(y[i]+e[i])-a.conj()*b/n)/(U-w[i]*abs(z[i])**2-abs(a)**2/n)
        assert np.allclose(reduced,fit(z[keep],(y+e)[keep],w[keep])[0])
    check(3, True)
    check(4, np.isclose(abs((50+.5)/.99-50), 1/.99) and 1/.99 < 1.011)
    check(5, (2/40**2)/(2/20**2) == .25)
    check(6, np.isclose((20*.1 + 10*.2)/127.5,4/127.5))

    for width,height in rng.integers(1,4096,size=(30,2)).tolist():
        caps = [min(32*c/width,32*(400//c)/height) for c in range(1,41)]
        i = next((c for c in range(1,41) if c*height >= (400//c)*width),40)
        assert max(caps) == max(caps[max(0,i-2):i])
    check(7, True)
    check(8, 36*832/1920 < 16 <= 37*832/1920)
    check(9, all(any(t <= x and x+96 <= t+640 for t in (0,544,1088)) for x in range(1633)))
    check(10, max(sum(g*b for g,b in zip((10,8,8),s)) for s in product((0,1),repeat=3)
                  if sum(c*b for c,b in zip((6,5,5),s))<=10)==16)
    d = 80*(1-.35)/(1+.35)
    check(11, np.isclose((80-d)/(80+d),.35))
    check(12, all(i/(140-i)<=.4 for i in range(41)))
    check(13, 131071*16384 <= 2**31-1 < 131072*16384)
    check(14, np.isclose(1.01**2-1,.0201))
    q,v = np.array([1,2]),np.array([.9,2.1])
    check(15, np.isclose(q@v/(q@q),1.02))
    A = rng.normal(size=(3,5))
    x,b,mu,beta = rng.normal(size=5),rng.normal(size=3),rng.normal(size=3),rng.normal(size=3)
    alpha = rng.normal(size=3)/np.sqrt(rng.uniform(.1,1,3)+1e-5)
    check(16, np.allclose(alpha*(A@x+b-mu)+beta,(alpha[:,None]*A)@x+alpha*(b-mu)+beta))
    check(17, 38*(64+256)==12160 and 64*256==16384)
    check(18, 1000*(4+3)-(120*4+20*3)==6460)

    vec = rng.normal(size=(7,10))
    vec /= np.linalg.norm(vec,axis=1,keepdims=True)
    p = np.arange(1,8)/28
    r = p@vec
    direction = r/np.linalg.norm(r)
    check(19, np.all(vec@r <= direction@r+1e-14))
    check(20, np.argmax(vec@r)==np.argmax((vec@vec.T)@p))
    distances = np.sum((vec[:,None]-vec)**2,axis=2)
    check(21, np.isclose(p@distances@p,2*(1-r@r)))
    check(22, np.isclose(1/np.dot([.8,.1,.1],[.8,.1,.1]), 1/.66))
    inv = np.linalg.solve(np.diag([1.,4.]),np.ones(2))
    check(23, np.allclose(inv/inv.sum(),[.8,.2]))
    check(24, round(math.degrees(math.asin(.25)),2)==14.48)
    check(25, np.allclose(np.sum((vec-direction)**2,axis=1),2-2*(vec@direction)))
    true = rng.uniform(-1,1,1000)
    rough = true + rng.uniform(-.01,.01,len(true))
    chk = np.argsort(-true)[:5]
    check(26, np.all(rough[chk]>=np.partition(rough,-5)[-5]-.02))
    check(27, .35+.4*.5<.56)
    check(28, math.cos(math.radians(60-15))<.75)
    check(29, .01131 < math.sqrt(512)*.001/2 < .01132)
    check(30, .81-.01>=.79 and .81-.76-.02>=.025)
    diffs = np.array([.8,.5])[:,None]-[.5,.2]
    check(31, np.mean(diffs>0)+.5*np.mean(diffs==0)==.875)
    pos,neg = np.array([.9,.8,.7]),np.array([.8,.8,.1,.1])
    check(32, max(sum(pos>=t) for t in (.1,.7,.8,.9,1) if sum(neg>=t)<=1)==1)
    check(33, .9999**29956 <= .05 < .9999**29955)
    check(34, .0951< -math.expm1(10000*math.log1p(-1e-5)) < .0952)
    check(35, 20*(1-.1)/.1==180)
    check(36, (990/1000+0)/2==.495 and 990/1001>.989)
    q0,q1 = vec[:2]
    check(37, np.max(np.abs(vec@(q1-q0))) <= np.linalg.norm(q1-q0))
    check(38, max(h for h in range(10) if 2*h<12)==5)
    costs = np.array([9.,1.])
    rates = np.sqrt(costs)/sum(np.sqrt(costs))
    check(39, np.allclose(rates,[.75,.25]) and sum(costs/rates)<sum(costs/.5))
    check(40, np.allclose([100+9/25*10,9*16/25],[103.6,5.76]))
    C = [[1,2],[2,100]]
    check(41, min(sum(C[i][j] for i,j in enumerate(s)) for s in permutations(range(2)))==4)
    check(42, np.isclose(.001+.999*(.01**3),.001000999))
    check(43, sum(t<250 for t in range(0,400,50))*2==10)
    check(44, 8+40+9*40==408)
    check(45, round(2/(.8*.020+.2*.100),2)==55.56)
    check(46, [min(20*k,12*(4-k)) for k in (1,2,3)]==[20,24,12])
    def feasible(indices):
        t=0
        for i in indices:
            t+=(4,2,2)[i]
            if t>(4,4,5)[i]:
                return False
        return True
    check(47, max(n for n in range(4) for ids in combinations(range(3),n) if feasible(ids))==2)
    check(48, 7/(2*40)==.0875)
    check(49, round(1/(.7+.3/4),2)==1.29)
    check(50, round(2*sum(math.comb(14,k) for k in range(3))/2**14,5)==.01294)
    check(51, math.ceil((1.96*2/.5)**2)==62)
    check(52, round(100/(1+2*sum((1-k/100)*.9**k for k in range(1,100))),2)==5.81)
    check(53, .95**59 <= .05 < .95**58)
    check(54, round(100*(1-.05**(1/15)),2)==18.10 and round(100*(1-.05**(1/3000)),4)==.0998)
    assert checked == list(range(1,55))
    print("Verified numerical examples for all 54 problems")


if __name__ == "__main__":
    verify()
