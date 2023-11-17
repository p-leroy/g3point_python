import numpy as np

from g3point_python.detrend import vec2rot

#%% first test

a = np.array([1, 0, 0])
b = np.array([0, 1, 0])

v = np.cross(a, b)
s = np.linalg.norm(v)  # sine of angle
c = np.dot(a, b)  # cosine of angle

G = np.array([[c, -s, 0],
              [s, c, 0],
              [0, 0, 1]])

Fi = np.c_[a, (b - c * a) / np.linalg.norm(b - c * a), np.cross(b,a)]

U = Fi @ G @ np.linalg.inv(Fi)

print(f'is it length-preserving? {np.linalg.norm(U, ord=2)} (expected answer 1)')  # is it length-preserving?
print(f'does it rotate a onto b? {np.linalg.norm(b - U @ a)} (expected answer 0)')  # does it rotate a onto b?

#%% with random values

a = np.random.rand(3)
b = np.random.rand(3)
a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)

v = np.cross(a, b)
s = np.linalg.norm(v)  # sine of angle
c = np.dot(a, b)  # cosine of angle

G = np.array([[c, -s, 0],
              [s, c, 0],
              [0, 0, 1]])

Fi = np.c_[a, (b - c * a) / np.linalg.norm(b - c * a), np.cross(b,a)]

U = Fi @ G @ np.linalg.inv(Fi)

print(f'is it length-preserving? {np.linalg.norm(U, ord=2)} (expected answer 1)')  # is it length-preserving?
print(f'does it rotate a onto b? {np.linalg.norm(b - U @ a)} (expected answer 0)')  # does it rotate a onto b?