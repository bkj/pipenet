# >>

# actions = np.random.choice((0, 1), (8, len(worker.pipes)))
# _ = env.step(actions)

# z = [str(bin(i)).split('b')[-1] for i in range(2 ** len(worker.pipes))]
# z = ['0' * (6 - len(zz)) + zz for zz in z]
# z = np.vstack(map(list, z)).astype(int)

# payout_list = []
# for _ in tqdm(range(72)):
#     _, payout, _, _ = env.step(z)
#     payout_list.append(payout.squeeze())

# payouts = np.vstack(payout_list).mean(axis=0)

# zs = z[np.argsort(-payouts)]
# ps = payouts[np.argsort(-payouts)]

# zs[:10]
# ps[:10]

# # !! Upshot is the spread between models w/ and w/o connections is _very_ small
# # !! What is the norm of the real paths vs. the untrained branches?

# <<