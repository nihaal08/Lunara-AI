import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.gestational_diabetes.train import train as train_gdm
from ml.maternal_health.train import train as train_maternal
from ml.preeclampsia.train import train as train_preeclampsia


def main() -> None:
    print("=" * 60)
    print("Training Gestational Diabetes model...")
    print("=" * 60)
    train_gdm()

    print("\n" + "=" * 60)
    print("Training Maternal Health model...")
    print("=" * 60)
    train_maternal()

    print("\n" + "=" * 60)
    print("Training Preeclampsia model...")
    print("=" * 60)
    train_preeclampsia()

    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
