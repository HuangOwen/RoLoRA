from llamafactory.train.tuner import export_model


def main():
    export_model()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    export_model()


if __name__ == "__main__":
    main()
