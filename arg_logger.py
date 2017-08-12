def log_args(args, log_filename):
    with open(log_filename, 'w') as f:
        for arg, val in vars(args).items():
            print("%s: %s" % (arg, val), file=f)
