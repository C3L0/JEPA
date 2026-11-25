import numpy as np

from .data import WindowArrays


def main():
    # Load the existing validation dataset.
    dataset = WindowArrays.from_npz("data/m4/val_windows.npz")

    actions = dataset.action_sequence
    action_mask = dataset.action_mask
    states = dataset.state_window
    targets = dataset.target_state
    horizons = dataset.horizon.reshape(-1)

    print("Number of validation samples:", len(horizons))
    print("Action sequence shape:", actions.shape)

    # Calculate action statistics using valid (non-padded) entries only.
    valid_actions = actions[action_mask.astype(bool)]

    print("\nAction statistics (valid entries only):")
    print("Mean:", valid_actions.mean(axis=0))
    print("Std:", valid_actions.std(axis=0))
    print("Min:", valid_actions.min(axis=0))
    print("Max:", valid_actions.max(axis=0))

    # Compare final context state with future target state.
    state_changes = targets - states[:, -1, :]

    print("\nState change statistics:")
    print("Mean absolute change:", np.mean(np.abs(state_changes)))
    print("Std:", np.std(state_changes))

    print("\nPer-dimension action standard deviations:")
    print(valid_actions.std(axis=0))

    print("\nPer-dimension state change standard deviations:")
    print(np.std(state_changes, axis=0))

    # Persistence baseline: predict that the state does not change.
    persistence_prediction = states[:, -1, :]

    persistence_mse = np.mean((persistence_prediction - targets) ** 2)

    print("\nPersistence baseline:")
    print("MSE:", persistence_mse)

    print("\nMean squared state change per dimension:")
    print(np.mean(state_changes**2, axis=0))

    # Check action variation for each prediction horizon.
    print("\nStatistics by horizon:")
    for h in np.unique(horizons):
        indices = np.flatnonzero(horizons == h)
        print(
            f"Horizon {h}: "
            f"{len(indices)} samples, "
            f"mean absolute state change = "
            f"{np.mean(np.abs(state_changes[indices])):.6f}"
        )


if __name__ == "__main__":
    main()
