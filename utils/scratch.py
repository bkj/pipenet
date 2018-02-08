# >>

all_configs = [str(bin(i)).split('b')[-1] for i in range(2 ** len(worker.pipes))]
all_configs = ['0' * (6 - len(config)) + config for config in all_configs]
all_configs = np.vstack(map(list, all_configs)).astype(int)


payouts = []
all_pipes = np.array(list(worker.pipes.keys()))
for config in all_configs:
    mask_pipes = [tuple(pipe) for pipe in all_pipes[config == 1]]
    worker.set_pipes(mask_pipes)
    try:
        acc = worker.eval_epoch(dataloaders, mode='test')
    except AccumulateException:
        acc = 0.0
    except:
        raise
    
    print(config, acc)
    payouts.append(acc)

all_configs[np.argsort(payouts)[::-1]]

# <<