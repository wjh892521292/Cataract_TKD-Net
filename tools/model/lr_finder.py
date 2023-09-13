from utils import CLI


def main():
    cli = CLI(run = False)
    # Run learning rate finder
    lr_finder = cli.trainer.tuner.lr_find(model = cli.model, datamodule = cli.datamodule)

    # Plot with
    fig = lr_finder.plot(suggest = True)
    fig.savefig('lr_finder.png')

    # Pick point based on plot, or get suggestion
    print(f'suggest lr: {lr_finder.suggestion()}')


if __name__ == '__main__':
    main()
