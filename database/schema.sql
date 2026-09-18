CREATE SCHEMA IF NOT EXISTS belt;

-- ==========================================
-- MACHINE
-- ==========================================

CREATE TABLE IF NOT EXISTS belt.machine (
    machine_id SERIAL PRIMARY KEY,
    machine_code VARCHAR(30) UNIQUE NOT NULL,
    machine_name VARCHAR(100),
    location VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ==========================================
-- CAMERA
-- ==========================================

CREATE TABLE IF NOT EXISTS belt.camera (
    camera_id SERIAL PRIMARY KEY,
    machine_id INTEGER REFERENCES belt.machine(machine_id),
    camera_name VARCHAR(100) NOT NULL,
    serial_number VARCHAR(100),
    ip_address VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ==========================================
-- MODEL VERSION
-- ==========================================

CREATE TABLE IF NOT EXISTS belt.model_version (
    model_id SERIAL PRIMARY KEY,

    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    model_type VARCHAR(50),
    model_path TEXT,

    confidence_threshold NUMERIC(5,4),

    precision NUMERIC(6,4),
    recall NUMERIC(6,4),
    map50 NUMERIC(6,4),
    map50_95 NUMERIC(6,4),

    status VARCHAR(20) DEFAULT 'STAGING',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(model_name, model_version)
);


-- ==========================================
-- IMAGE
-- ==========================================

CREATE TABLE IF NOT EXISTS belt.image (
    image_id BIGSERIAL PRIMARY KEY,

    camera_id INTEGER REFERENCES belt.camera(camera_id),

    filename TEXT NOT NULL,
    image_path TEXT NOT NULL,
    result_path TEXT,

    capture_time TIMESTAMP,

    status VARCHAR(20) NOT NULL DEFAULT 'WAITING',

    processed_time TIMESTAMP,

    inference_time_ms NUMERIC(8,2),

    error_message TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ==========================================
-- DETECTION
-- ==========================================

CREATE TABLE IF NOT EXISTS belt.detection (
    detection_id BIGSERIAL PRIMARY KEY,

    image_id BIGINT NOT NULL
        REFERENCES belt.image(image_id)
        ON DELETE CASCADE,

    model_id INTEGER REFERENCES belt.model_version(model_id),

    class_id INTEGER NOT NULL,
    class_name VARCHAR(50) NOT NULL,

    confidence NUMERIC(6,5) NOT NULL,

    x1 NUMERIC(10,2),
    y1 NUMERIC(10,2),
    x2 NUMERIC(10,2),
    y2 NUMERIC(10,2),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE belt.inspection_result (
    inspection_id BIGSERIAL PRIMARY KEY,
    image_id BIGINT NOT NULL,
    roll_detected BOOLEAN NOT NULL DEFAULT FALSE,
    belt1_detected BOOLEAN NOT NULL DEFAULT FALSE,
    belt2_detected BOOLEAN NOT NULL DEFAULT FALSE,
    good_count INTEGER NOT NULL DEFAULT 0,
    splice_count INTEGER NOT NULL DEFAULT 0,
    dogear_count INTEGER NOT NULL DEFAULT 0,
    overall_result VARCHAR(10) NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT inspection_result_image_id_fkey
        FOREIGN KEY (image_id)
        REFERENCES belt.image(image_id)
        ON DELETE CASCADE,

    CONSTRAINT inspection_result_image_id_key
        UNIQUE (image_id),

    CONSTRAINT inspection_result_check
        CHECK (overall_result IN ('GOOD', 'NG'))
);


-- ==========================================
-- INDEX
-- ==========================================

CREATE INDEX IF NOT EXISTS idx_camera_machine
    ON belt.camera(machine_id);

CREATE INDEX IF NOT EXISTS idx_image_camera
    ON belt.image(camera_id);

CREATE INDEX IF NOT EXISTS idx_image_capture_time
    ON belt.image(capture_time);

CREATE INDEX IF NOT EXISTS idx_detection_image
    ON belt.detection(image_id);

CREATE INDEX IF NOT EXISTS idx_detection_model
    ON belt.detection(model_id);

CREATE INDEX IF NOT EXISTS idx_model_status
    ON belt.model_version(status);