from arg_parser import parse_args
from lagan import LaGAN

"""main"""


def main():
  # parse arguments
  args = parse_args()
  if args is None:
    exit()

  gan = LaGAN(args)

  if args.phase == 'train':
    gan.train()
    print("[*] training finished!")

  if args.phase == 'test':
    gan.test()
    print("[*] test finished!")

  if args.phase == 'translate':
    gan.translate()
    print("[*] translation finished!")


if __name__ == '__main__':
  main()
