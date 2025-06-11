import random
from math import pi, cos, sin
import numpy as np
from sklearn.datasets._samples_generator import make_blobs

F = random.uniform(0.0, 1.0)

#explain: this needs to be fix when generating default ones
F = 1.0

def m_rot(a):
    return np.matrix([[cos(a), -sin(a), 0],
                      [sin(a), cos(a), 0],
                      [0, 0, 1]])


def m_trans(x, y):
    return np.matrix([[1, 0, x],
                      [0, 1, y],
                      [0, 0, 1]])


def m_scale(sx, sy):
    return np.matrix([[sx, 0, 0],
                      [0, sy, 0],
                      [0, 0, 1]])


def random_translate(max_dist):
    theta = random.uniform(-pi, pi)
    norm = random.uniform(0, max_dist) * F
    tx, ty = norm * cos(theta), norm * sin(theta)

    return m_trans(tx, ty)
def random_global_translate(min_x, max_x, min_y, max_y):
    theta = random.uniform(-pi, pi)
    tx = random.uniform(min_x, max_x)
    ty = random.uniform(min_y, max_y)


    return m_trans(tx, ty)

def random_scale(minimum, maximum, par):
    minimum = 1 + (minimum - 1) * F
    maximum = 1 + (maximum - 1) * F

    if par:
        r1 = random.uniform(minimum, maximum)
        r2 = r1
    else:
        r1 = random.uniform(minimum, maximum)
        r2 = random.uniform(minimum, maximum)

    return m_scale(r1, r2)


def getRandomTransform(center, params):
    MAX_ANGLE = params['MAX_ANGLE']
    MIN_SCALE = params['MIN_SCALE']
    MAX_SCALE = params['MAX_SCALE']
    MAX_TRANSLATE = params['MAX_TRANSLATE']
    PRESERVE_RATIO = params['PRESERVE_RATIO']

    T_ori = m_trans(-center[0], -center[1])

    a = random.uniform(-MAX_ANGLE, MAX_ANGLE) * F

    R = m_rot(a)
    S = random_scale(MIN_SCALE, MAX_SCALE, PRESERVE_RATIO)
    T_inv = m_trans(center[0], center[1])
    T = random_translate(MAX_TRANSLATE)
    M = T * T_inv * S * R * T_ori
    return M


def getGlobalTransform(center, params):
    """
    function: there is to get a random transform to rotate all the paths in the svg together:
    i include the rotation angle distribution within the function, which makes it not general enough.
    :param center:
    :type center:
    :param params:
    :type params:
    :return:
    :rtype:
    """
    MAX_ANGLE = params['MAX_ANGLE']
    MIN_SCALE = params['MIN_SCALE']
    MAX_SCALE = params['MAX_SCALE']
    MAX_TRANSLATE_X = params['MAX_TRANSLATE_X']
    MIN_TRANSLATE_X = params['MIN_TRANSLATE_X']
    MAX_TRANSLATE_Y = params['MAX_TRANSLATE_Y']
    MIN_TRANSLATE_Y = params['MIN_TRANSLATE_Y']
    PRESERVE_RATIO = params['PRESERVE_RATIO']

    T_ori = m_trans(-center[0], -center[1])

    a = random.uniform(-MAX_ANGLE, MAX_ANGLE)
    R = m_rot(a)


    if PRESERVE_RATIO:
        r1 = random.uniform(MIN_SCALE, MAX_SCALE)
        r2 = r1
    else:
        r1 = random.uniform(MIN_SCALE, MAX_SCALE)
        r2 = random.uniform(MIN_SCALE, MAX_SCALE)

    S = m_scale(r1, r2)


    T_inv = m_trans(center[0], center[1])
    T = random_global_translate(MIN_TRANSLATE_X, MAX_TRANSLATE_X, MIN_TRANSLATE_Y, MAX_TRANSLATE_Y)

    M = T * T_inv * S * R * T_ori



    return M

def getGlobalTransform_Oneside(center, params):
    """
    function: there is to get a random transform to rotate all the paths in the svg together, but only 0 to - centain angle not including 0 to centain angle:
    i include the rotation angle distribution within the function, which makes it not general enough.
    :param center:
    :type center:
    :param params:
    :type params:
    :return:
    :rtype:
    """
    MAX_ANGLE = params['MAX_ANGLE']
    MIN_SCALE = params['MIN_SCALE']
    MAX_SCALE = params['MAX_SCALE']
    MAX_TRANSLATE_X = params['MAX_TRANSLATE_X']
    MIN_TRANSLATE_X = params['MIN_TRANSLATE_X']
    MAX_TRANSLATE_Y = params['MAX_TRANSLATE_Y']
    MIN_TRANSLATE_Y = params['MIN_TRANSLATE_Y']
    PRESERVE_RATIO = params['PRESERVE_RATIO']

    T_ori = m_trans(-center[0], -center[1])

    #a = random.uniform(-MAX_ANGLE, MAX_ANGLE)
    a = random.uniform(-MAX_ANGLE, 0)

    R = m_rot(a)


    if PRESERVE_RATIO:
        r1 = random.uniform(MIN_SCALE, MAX_SCALE)
        r2 = r1
    else:
        r1 = random.uniform(MIN_SCALE, MAX_SCALE)
        r2 = random.uniform(MIN_SCALE, MAX_SCALE)

    S = m_scale(r1, r2)


    T_inv = m_trans(center[0], center[1])
    T = random_global_translate(MIN_TRANSLATE_X, MAX_TRANSLATE_X, MIN_TRANSLATE_Y, MAX_TRANSLATE_Y)

    M = T * T_inv * S * R * T_ori



    return M
def getRandomGlobalTransform(center, params):
    """
    function: there is to get a random transform to rotate all the paths in the svg together:
    i include the rotation angle distribution within the function, which makes it not general enough.
    :param center:
    :type center:
    :param params:
    :type params:
    :return:
    :rtype:
    """
    MAX_ANGLE = params['MAX_ANGLE']
    MIN_SCALE = params['MIN_SCALE']
    MAX_SCALE = params['MAX_SCALE']
    MAX_TRANSLATE_X = params['MAX_TRANSLATE_X']
    MIN_TRANSLATE_X = params['MIN_TRANSLATE_X']
    MAX_TRANSLATE_Y = params['MAX_TRANSLATE_Y']
    MIN_TRANSLATE_Y = params['MIN_TRANSLATE_Y']
    PRESERVE_RATIO = params['PRESERVE_RATIO']

    T_ori = m_trans(-center[0], -center[1])

    a = random.uniform(-MAX_ANGLE, MAX_ANGLE)
    R = m_rot(a)
    S = random_scale(MIN_SCALE, MAX_SCALE, PRESERVE_RATIO)
    T_inv = m_trans(center[0], center[1])
    T = random_global_translate(MIN_TRANSLATE_X, MAX_TRANSLATE_X, MIN_TRANSLATE_Y, MAX_TRANSLATE_Y)

    M = T * T_inv * S * R * T_ori



    return M

def getRandomGlobalTransform_distribtion(center, params):
    """
    function: there is to get a random transform to rotate all the paths in the svg together:
    i include the rotation angle distribution within the function, which makes it not general enough.
    :param center:
    :type center:
    :param params:
    :type params:
    :return:
    :rtype:
    """
    MAX_ANGLE = params['MAX_ANGLE']
    MIN_SCALE = params['MIN_SCALE']
    MAX_SCALE = params['MAX_SCALE']
    MAX_TRANSLATE_X = params['MAX_TRANSLATE_X']
    MIN_TRANSLATE_X = params['MIN_TRANSLATE_X']
    MAX_TRANSLATE_Y = params['MAX_TRANSLATE_Y']
    MIN_TRANSLATE_Y = params['MIN_TRANSLATE_Y']
    PRESERVE_RATIO = params['PRESERVE_RATIO']

    T_ori = m_trans(-center[0], -center[1])

    #a = random.uniform(-MAX_ANGLE, MAX_ANGLE)
    # explain: if only generate one, then it would be in center 0, not sure why. So generate 100 samples and select the first one
    a, y_true = make_blobs(n_samples=100, n_features=1, centers=np.array([0, 1.5708, 3.1415]).reshape(-1, 1), cluster_std=0.20, random_state=None)
    print(a[0,0])
    R = m_rot(a[0,0])
    S = random_scale(MIN_SCALE, MAX_SCALE, PRESERVE_RATIO)
    T_inv = m_trans(center[0], center[1])
    T = random_global_translate(MIN_TRANSLATE_X, MAX_TRANSLATE_X, MIN_TRANSLATE_Y, MAX_TRANSLATE_Y)

    M = T * T_inv * S * R * T_ori


    return M


def getRandomGlobalTransform_value(center, params, rotation_value):
    """
    function: there is to get a random transform to rotate all the paths in the svg together:
    i include the rotation angle distribution within the function, which makes it not general enough.
    :param center:
    :type center:
    :param params:
    :type params:
    :return:
    :rtype:
    """
    MAX_ANGLE = params['MAX_ANGLE']
    MIN_SCALE = params['MIN_SCALE']
    MAX_SCALE = params['MAX_SCALE']
    MAX_TRANSLATE_X = params['MAX_TRANSLATE_X']
    MIN_TRANSLATE_X = params['MIN_TRANSLATE_X']
    MAX_TRANSLATE_Y = params['MAX_TRANSLATE_Y']
    MIN_TRANSLATE_Y = params['MIN_TRANSLATE_Y']
    PRESERVE_RATIO = params['PRESERVE_RATIO']

    T_ori = m_trans(-center[0], -center[1])

    a = random.uniform(rotation_value-0.1, rotation_value+0.1)

    R = m_rot(a)
    S = random_scale(MIN_SCALE, MAX_SCALE, PRESERVE_RATIO)
    T_inv = m_trans(center[0], center[1])
    T = random_global_translate(MIN_TRANSLATE_X, MAX_TRANSLATE_X, MIN_TRANSLATE_Y, MAX_TRANSLATE_Y)

    M = T * T_inv * S * R * T_ori


    return M

def getRandomGlobalTransform_triangle(center, params):
    """
    function: there is to get a random transform to rotate all the paths in the svg together:
    i include the rotation angle distribution within the function, which makes it not general enough.
    :param center:
    :type center:
    :param params:
    :type params:
    :return:
    :rtype:
    """
    MAX_ANGLE = params['MAX_ANGLE']
    MIN_SCALE = params['MIN_SCALE']
    MAX_SCALE = params['MAX_SCALE']
    MAX_TRANSLATE_X = params['MAX_TRANSLATE_X']
    MIN_TRANSLATE_X = params['MIN_TRANSLATE_X']
    MAX_TRANSLATE_Y = params['MAX_TRANSLATE_Y']
    MIN_TRANSLATE_Y = params['MIN_TRANSLATE_Y']
    PRESERVE_RATIO = params['PRESERVE_RATIO']

    REFLECT = params['REFLECT']

    T_ori = m_trans(-center[0], -center[1])
    a = random.uniform(-MAX_ANGLE, MAX_ANGLE)
    R = m_rot(a)
    S = random_scale(MIN_SCALE, MAX_SCALE, PRESERVE_RATIO)
    T_inv = m_trans(center[0], center[1])
    T = random_global_translate(MIN_TRANSLATE_X, MAX_TRANSLATE_X, MIN_TRANSLATE_Y, MAX_TRANSLATE_Y)

    if REFLECT == 'X':
        REF = np.array([[-1,0,0], [0,1,0], [0,0,1]])
    elif REFLECT == 'Y':
        REF = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    elif REFLECT == 'XY':
        REF = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    else:
        REF = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    M = T * T_inv * S * REF * R * T_ori
    return M





def disturbPoly(poly, M_local=np.identity(3), noise=0):
    points = []
    for v in poly['points']:
        P = np.matrix([v[0], v[1], 1]).transpose()
        T_noise = random_translate(noise)
        p_rot = T_noise * M_local * P
        points.append([p_rot.item(0), p_rot.item(1)])
    return points


def coherentDisturb(a, b, p):
    u = np.array(b) - np.array(a)
    v = np.array([-u[1], u[0]])
    if np.linalg.norm(v) != 0:
        v = v / np.linalg.norm(v)
    return (b + random.uniform(-p, p) * F * v).tolist()


def addOverstroke(data, p, under=False):
    b, c = np.array(data[0]), np.array(data[1])
    y, x = np.array(data[len(data) - 1]), np.array(data[len(data) - 2])

    u = b - c;
    nu = np.linalg.norm(u)
    if nu > 0:
        u = u / nu
    else:
        u = np.array([0, 0])
    r = random.uniform(max(-p, -nu) if under else 0, p) * F * 2.0
    #explain: 3.0 is to make a longer overstroke, but sometimes, the new point can be overlapped with the existing points, cause problem in Path.point(t)




    # print b, "+", r, u
    a = b + r * u

    v = y - x
    nv = np.linalg.norm(v)
    if nv > 0:
        v = v / nv
    else:
        v = np.array([0, 0])
    s = random.uniform(max(-p, -nv) if under else 0, p) * F
    # print y, "+", s, v
    z = y + s * v

    if r > 0:
        data.insert(0, [a[0], a[1]])
    else:
        data[0] = [a[0], a[1]]

    if s > 0:
        data.append([z[0], z[1]])
    else:
        data[len(data) - 1] = [z[0], z[1]]
