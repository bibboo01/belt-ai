from pathlib import Path
import json
import logging

from ultralytics import YOLO


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PRODUCTION_DIR = (
    BASE_DIR
    / "models"
    / "production"
)

MODEL_POINTER = (
    PRODUCTION_DIR
    / "model.json"
)


# ==========================================================
# Logging
# ==========================================================

logger = logging.getLogger("belt_ai")


# ==========================================================
# Expected Model Classes
# ==========================================================

EXPECTED_CLASSES = {
    0: "dogear-belt",
    1: "good-belt",
    2: "splice-belt",
}


# ==========================================================
# Model Manager
# ==========================================================

class ModelManager:

    def __init__(self):

        self.model = None

        self.model_id = None
        self.model_name = None
        self.model_version = None

        self.confidence = 0.5

        self.status = "NOT_LOADED"


    # ======================================================
    # Reset State
    # ======================================================

    def _reset_state(self):

        self.model = None

        self.model_id = None
        self.model_name = None
        self.model_version = None

        self.status = "NOT_LOADED"


    # ======================================================
    # Load Model
    # ======================================================

    def load_model(self):

        # --------------------------------------------------
        # Reset previous state
        # --------------------------------------------------

        self._reset_state()


        # --------------------------------------------------
        # 1. Check model pointer
        # --------------------------------------------------

        if not MODEL_POINTER.exists():

            self.status = (
                "NO_PRODUCTION_MODEL"
            )

            logger.error(
                "Model pointer not found | path=%s",
                MODEL_POINTER,
            )

            return False


        # --------------------------------------------------
        # 2. Read model.json
        # --------------------------------------------------

        try:

            with open(
                MODEL_POINTER,
                "r",
                encoding="utf-8-sig",
            ) as f:

                config = json.load(f)

        except json.JSONDecodeError:

            self.status = (
                "MODEL_CONFIG_INVALID"
            )

            logger.exception(
                "Invalid model.json | path=%s",
                MODEL_POINTER,
            )

            return False

        except OSError:

            self.status = (
                "MODEL_CONFIG_READ_FAILED"
            )

            logger.exception(
                "Unable to read model.json | path=%s",
                MODEL_POINTER,
            )

            return False


        # --------------------------------------------------
        # 3. Validate config type
        # --------------------------------------------------

        if not isinstance(
            config,
            dict,
        ):

            self.status = (
                "MODEL_CONFIG_INVALID"
            )

            logger.error(
                "Model configuration must be a JSON object"
            )

            return False


        # --------------------------------------------------
        # 4. Read metadata
        # --------------------------------------------------

        self.model_id = config.get(
            "model_id"
        )

        self.model_name = config.get(
            "model_name"
        )

        self.model_version = config.get(
            "model_version"
        )

        model_file = config.get(
            "model_file"
        )


        # --------------------------------------------------
        # 5. Validate model_id
        # --------------------------------------------------

        if self.model_id is None:

            self.status = (
                "MODEL_CONFIG_INVALID"
            )

            logger.error(
                "model_id missing from model.json"
            )

            return False


        # --------------------------------------------------
        # 6. Validate model_name
        # --------------------------------------------------

        if (
            not isinstance(
                self.model_name,
                str,
            )
            or not self.model_name.strip()
        ):

            self.status = (
                "MODEL_CONFIG_INVALID"
            )

            logger.error(
                "model_name missing from model.json"
            )

            return False


        # --------------------------------------------------
        # 7. Validate model_version
        # --------------------------------------------------

        if (
            not isinstance(
                self.model_version,
                str,
            )
            or not self.model_version.strip()
        ):

            self.status = (
                "MODEL_CONFIG_INVALID"
            )

            logger.error(
                "model_version missing from model.json"
            )

            return False


        # --------------------------------------------------
        # 8. Validate model_file
        # --------------------------------------------------

        if (
            not isinstance(
                model_file,
                str,
            )
            or not model_file.strip()
        ):

            self.status = (
                "NO_PRODUCTION_MODEL"
            )

            logger.error(
                "model_file missing from model.json"
            )

            return False


        # --------------------------------------------------
        # 9. Prevent Path Traversal
        # --------------------------------------------------

        try:

            production_root = (
                PRODUCTION_DIR
                .resolve()
            )

            model_path = (
                PRODUCTION_DIR
                / model_file
            ).resolve()

            model_path.relative_to(
                production_root
            )

        except ValueError:

            self.status = (
                "INVALID_MODEL_PATH"
            )

            logger.error(
                "Model path outside production directory | "
                "model_file=%s",
                model_file,
            )

            return False


        # --------------------------------------------------
        # 10. Check model file exists
        # --------------------------------------------------

        if not model_path.exists():

            self.status = (
                "MODEL_FILE_NOT_FOUND"
            )

            logger.error(
                "Model file not found | path=%s",
                model_path,
            )

            return False


        # --------------------------------------------------
        # 11. Check model path is file
        # --------------------------------------------------

        if not model_path.is_file():

            self.status = (
                "MODEL_FILE_INVALID"
            )

            logger.error(
                "Model path is not a file | path=%s",
                model_path,
            )

            return False


        # --------------------------------------------------
        # 12. Load YOLO
        # --------------------------------------------------

        try:

            loaded_model = YOLO(
                str(model_path)
            )

        except Exception:

            self.status = (
                "MODEL_LOAD_FAILED"
            )

            logger.exception(
                "Failed to load YOLO model | "
                "path=%s | "
                "model=%s | "
                "version=%s",
                model_path,
                self.model_name,
                self.model_version,
            )

            return False


        # --------------------------------------------------
        # 13. Validate task
        # --------------------------------------------------

        task = getattr(
            loaded_model,
            "task",
            None,
        )


        if task != "detect":

            self.status = (
                "MODEL_INVALID_TASK"
            )

            logger.error(
                "Invalid model task | "
                "expected=detect | "
                "actual=%s",
                task,
            )

            return False


        # --------------------------------------------------
        # 14. Validate class mapping
        # --------------------------------------------------

        names = getattr(
            loaded_model,
            "names",
            None,
        )


        if not isinstance(
            names,
            dict,
        ):

            self.status = (
                "MODEL_INVALID_CLASSES"
            )

            logger.error(
                "Model class mapping is invalid"
            )

            return False


        # --------------------------------------------------
        # Normalize class keys
        # --------------------------------------------------

        normalized_names = {}

        try:

            for key, value in names.items():

                normalized_names[
                    int(key)
                ] = str(value)

        except Exception:

            self.status = (
                "MODEL_INVALID_CLASSES"
            )

            logger.exception(
                "Unable to normalize model classes"
            )

            return False


        # --------------------------------------------------
        # Compare classes
        # --------------------------------------------------

        if (
            normalized_names
            != EXPECTED_CLASSES
        ):

            self.status = (
                "MODEL_INVALID_CLASSES"
            )

            logger.error(
                "Model classes do not match expected classes | "
                "expected=%s | "
                "actual=%s",
                EXPECTED_CLASSES,
                normalized_names,
            )

            return False


        # --------------------------------------------------
        # 15. Commit loaded model
        # --------------------------------------------------

        self.model = loaded_model

        self.status = "LOADED"


        logger.info(
            "Model loaded successfully | "
            "model_id=%s | "
            "model=%s | "
            "version=%s | "
            "path=%s",
            self.model_id,
            self.model_name,
            self.model_version,
            model_path,
        )


        return True


    # ======================================================
    # Get Model
    # ======================================================

    def get_model(self):

        if self.model is None:

            self.load_model()

        return self.model


    # ======================================================
    # Get Info
    # ======================================================

    def get_info(self):

        return {

            "model_id": self.model_id,

            "model_name": self.model_name,

            "model_version": self.model_version,

            "status": self.status,

        }


    # ======================================================
    # Is Ready
    # ======================================================

    def is_ready(self):

        return (

            self.model is not None

            and

            self.status == "LOADED"

        )


# ==========================================================
# Singleton
# ==========================================================

model_manager = ModelManager()


# ==========================================================
# Manual Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BELT AI - MODEL MANAGER")
    print("=" * 70)

    loaded = (
        model_manager.load_model()
    )

    print(
        f"Loaded : {loaded}"
    )

    print(
        f"Info   : "
        f"{model_manager.get_info()}"
    )

    print("=" * 70)
