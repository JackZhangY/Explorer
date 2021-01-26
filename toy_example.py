

# def Faustian_model1():
#     '''model 1 with fixed V2'''
#     # hyperparameter
#     gamma = 0.99
#     alpha = 0.9
#     v_s2 = -1.8
#
#
#     # initial q value for state 1, the optimal q should satisfy q(s1, a1) > q(s1, a2)
#     q_s1_a1 = 0
#     q_s1_a2 = 1
#
#     q_s1_a1_AL = 0
#     q_s1_a2_AL = 1
#
#     q_s1_a1_clipAL = 0
#     q_s1_a2_clipAL = 1
#
#     # calculate the q value under state 1 for k steps
#     t = 0
#     while True:
#         v_s1_clipAL = max(q_s1_a1_clipAL, q_s1_a2_clipAL)
#         v_s1_AL = max(q_s1_a1_AL, q_s1_a2_AL)
#         v_s1 = max(q_s1_a1, q_s1_a2)
#
#         q_s1_a1 = 0 + gamma * v_s1
#         q_s1_a2 = 1 + gamma * (0.5 * v_s1 + 0.5 * v_s2)
#
#         q_s1_a1_AL = 0 + gamma * v_s1_AL - alpha * (v_s1_AL - q_s1_a1_AL)
#         q_s1_a2_AL = 1 + gamma * (0.5 * v_s1_AL + 0.5 * v_s2)- alpha * (v_s1_AL -q_s1_a2_AL)
#
#         mask_s1_a1 = float(q_s1_a1_clipAL / v_s1_clipAL > 0.8)
#         mask_s1_a2 = float(q_s1_a2_clipAL / v_s1_clipAL > 0.8)
#         q_s1_a1_clipAL = 0 + gamma * v_s1_clipAL - alpha * (v_s1_clipAL - q_s1_a1_clipAL) * mask_s1_a1
#         q_s1_a2_clipAL = 1 + gamma * (0.5 * v_s1_clipAL + 0.5 * v_s2) - alpha * (v_s1_clipAL -q_s1_a2_clipAL) * mask_s1_a2
#
#         print('opt:({},{}), AL: ({}, {}), clipAL:({}, {})'.format(q_s1_a1, q_s1_a2, q_s1_a1_AL, q_s1_a2_AL, q_s1_a1_clipAL,
#                                                                   q_s1_a2_clipAL))
#         # if q_s1_a1 > q_s1_a2 or q_s1_a1_AL > q_s1_a2_AL or t > 1000:
#         #     break
#         if t > 1000:
#             break
#         t += 1

def Faustian_model2():
    '''model 2 without fixed V2'''
    # hyperparameter
    gamma = 0.99
    alpha = 0.9
    clipratio = 0.8
    R_S1_A1 = 1
    R_S1_A2 = 2
    R_S2_A1 = -0

    # initial action value
    # the error order under in state 1
    INIT_S1_A1 = 1
    INIT_S1_A2 = 2
    INIT_S2_A1 = 1
    FLAG_OPT, FLAG_AL, FLAG_CLIPAL = True, True, True

    q_s1_a1 = INIT_S1_A1
    q_s1_a2 = INIT_S1_A2
    q_s2_a1 = INIT_S2_A1

    q_s1_a1_AL = INIT_S1_A1
    q_s1_a2_AL = INIT_S1_A2
    q_s2_a1_AL = INIT_S2_A1

    q_s1_a1_clipAL = INIT_S1_A1
    q_s1_a2_clipAL = INIT_S1_A2
    q_s2_a1_clipAL = INIT_S2_A1

    # the value iteration

    t = 0
    while True:
        # bellman optimal operator:
        v_s1 = max(q_s1_a1, q_s1_a2)
        v_s2 = q_s2_a1

        q_s1_a1 = R_S1_A1 + gamma * v_s1
        q_s1_a2 = R_S1_A2 + gamma * (0.5 * v_s1 + 0.5 * v_s2)

        q_s2_a1 = R_S2_A1 + gamma * v_s2

        # Advantage learning operator:
        v_s1_AL = max(q_s1_a1_AL, q_s1_a2_AL)
        v_s2_AL = q_s2_a1_AL

        q_s1_a1_AL = R_S1_A1 + gamma * v_s1_AL - alpha * (v_s1_AL - q_s1_a1_AL)
        q_s1_a2_AL = R_S1_A2 + gamma * (0.5 * v_s1_AL + 0.5 * v_s2_AL) - alpha * (v_s1_AL - q_s1_a2_AL)

        q_s2_a1_AL = R_S2_A1 + gamma * v_s2_AL

        # clip Advantage learning operator:
        v_s1_clipAL = max(q_s1_a1_clipAL, q_s1_a2_clipAL)
        v_s2_clipAL = q_s2_a1_clipAL


        mask_a1 = float(q_s1_a1_clipAL / v_s1_clipAL > clipratio)
        mask_a2 = float(q_s1_a2_clipAL / v_s1_clipAL > clipratio)
        q_s1_a1_clipAL = R_S1_A1 + gamma * v_s1_clipAL - alpha * (v_s1_clipAL -q_s1_a1_clipAL) * mask_a1
        q_s1_a2_clipAL = R_S1_A2 + gamma * (0.5 * v_s1_clipAL + 0.5 * v_s2_clipAL) - alpha * (v_s1_clipAL - q_s1_a2_clipAL) * mask_a2

        q_s2_a1_clipAL = R_S2_A1 + gamma * v_s2_clipAL

        if q_s1_a1 > q_s1_a2 and FLAG_OPT:
            print('##### bellman alternation #####')
            FLAG_OPT = False
        if q_s1_a1_AL > q_s1_a2_AL and FLAG_AL:
            print('##### advantage learning alternation #####')
            FLAG_AL = False
        if q_s1_a1_clipAL > q_s1_a2_clipAL and FLAG_CLIPAL:
            print('##### clip advantage learning alternation #####')
            FLAG_CLIPAL = False
        print('opt:({:.6f}, {:.6f}), ({:.6f}), al:({:.6f}, {:.6f}), ({:.6f}), clipal:({:.6f}, {:.6f}), ({:.6f}),'.format(
            q_s1_a1, q_s1_a2, q_s2_a1, q_s1_a1_AL, q_s1_a2_AL, q_s2_a1_AL, q_s1_a1_clipAL, q_s1_a2_clipAL, q_s2_a1_clipAL))

        if t > 1000:
            break
        t += 1

def underexplore(qs1a1=0, qs1a2=0, qs2a1=0):
    # hyperparameter
    gamma = 0.99
    alpha = 0.9
    clipratio = 0.90

    R_S1_A1 = 1
    R_S1_A2 = 0
    R_S2_A1 = 2

    # initial action value
    # the error order under in state 1 50, 0, 0
    INIT_S1_A1 = qs1a1
    INIT_S1_A2 = qs1a2
    INIT_S2_A1 = qs2a1
    FLAG_OPT, FLAG_AL, FLAG_CLIPAL = True, True, True
    epsilon = 0

    q_s1_a1 = INIT_S1_A1
    q_s1_a2 = INIT_S1_A2
    q_s2_a1 = INIT_S2_A1

    q_s1_a1_AL = INIT_S1_A1
    q_s1_a2_AL = INIT_S1_A2
    q_s2_a1_AL = INIT_S2_A1

    q_s1_a1_clipAL = INIT_S1_A1
    q_s1_a2_clipAL = INIT_S1_A2
    q_s2_a1_clipAL = INIT_S2_A1

    t = 0
    while True:

        # bellman
        v_s1 = max(q_s1_a1, q_s1_a2)
        v_s2 = q_s2_a1

        q_s1_a1 = R_S1_A1 + gamma * v_s1
        q_s1_a2 = R_S1_A2 + gamma * v_s2
        q_s2_a1 = R_S2_A1 + gamma * v_s2

        # al
        v_s1_AL = max(q_s1_a1_AL, q_s1_a2_AL)
        v_s2_AL = q_s2_a1_AL

        q_s1_a1_AL = R_S1_A1 + gamma * v_s1_AL - alpha * (v_s1_AL - q_s1_a1_AL)
        q_s1_a2_AL = R_S1_A2 + gamma * v_s2_AL - alpha * (v_s1_AL - q_s1_a2_AL)
        q_s2_a1_AL = R_S2_A1 + gamma * v_s2_AL

        # clip al
        v_s1_clipAL = max(q_s1_a1_clipAL, q_s1_a2_clipAL)
        v_s2_clipAL = q_s2_a1_clipAL

        mask1 = float((q_s1_a1_clipAL / (v_s1_clipAL+epsilon)) > clipratio)
        mask2 = float((q_s1_a2_clipAL / (v_s1_clipAL+epsilon)) > clipratio)

        q_s1_a1_clipAL = R_S1_A1 + gamma * v_s1_clipAL - alpha * (v_s1_clipAL - q_s1_a1_clipAL) * mask1
        q_s1_a2_clipAL = R_S1_A2 + gamma * v_s2_clipAL - alpha * (v_s1_clipAL - q_s1_a2_clipAL) * mask2
        q_s2_a1_clipAL = R_S2_A1 + gamma * v_s2_clipAL

        if q_s1_a1 < q_s1_a2 and FLAG_OPT:
            print('##### bellman alternation: {} steps #####'.format(t+1))
            FLAG_OPT = False
        if q_s1_a1_AL < q_s1_a2_AL and FLAG_AL:
            print('##### advantage learning alternation: {} steps #####'.format(t+1))
            FLAG_AL = False
        if q_s1_a1_clipAL < q_s1_a2_clipAL and FLAG_CLIPAL:
            print('##### clip advantage learning alternation: {} steps #####'.format(t+1))
            FLAG_CLIPAL = False
        print('opt:({:.6f}, {:.6f}), ({:.6f}), al:({:.6f}, {:.6f}), ({:.6f}), clipal:({:.6f}, {:.6f}), ({:.6f}),'.format(
            q_s1_a1, q_s1_a2, q_s2_a1, q_s1_a1_AL, q_s1_a2_AL, q_s2_a1_AL, q_s1_a1_clipAL, q_s1_a2_clipAL, q_s2_a1_clipAL))

        if t > 1000:
            break
        t += 1

if __name__ == '__main__':
    # Faustian_model2()
    underexplore(100, 1, 1)