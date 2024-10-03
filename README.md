# Drone Detection System with OOP and Audio Detection

This project uses YOLOV8 to detect drones from a camera feed and performs basic audio detection. It provides a GUI for starting, pausing, and stopping the detection, and displays detections, current alerts, and alert history.

## Features

- **Real-time Drone Detection**: Utilizes a YOLOV8 custom model to detect drones in live video feeds.
- **Audio Detection**: Monitors audio input for drone sounds.
- **Graphical User Interface**: User-friendly GUI built with Tkinter.
- **Multilingual Support**: Supports English and French languages.
- **Dark Mode**: Option to switch between light and dark themes.
- **Alert History**: Saves detection history with snapshots and exports to XML.
- **Settings Configuration**: Adjustable detection thresholds and other settings.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Directory Structure](#directory-structure)
- [Naming Conventions](#naming-conventions)
- [Backup and Version Control](#backup-and-version-control)
- [Data Structures](#data-structures)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- Python 3.8.19 required
- pip (Python package manager)
- See requirements.txt
- Conda environment (recommended for consistent use and avoiding errors after module updates)

Clone the Repository

```bash
git clone https://github.com/I am not going to publish this just proof of concept/drone-detection-system.git
cd drone-detection-system
```

### Set Up a Virtual Environment (Optional but Recommended to actually work)

```bash
python -m venv venv
# Activate the virtual environment:
# On Windows:
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Model Weights

- Download the YOLO model weights (`best.pt`) and place them in the `models/` directory. Keeping in mind to replace the backslash with fordward slash to avoid
- Ensure that the path to the model weights in your code matches the location.

### Configure Settings

- Copy `config/settings.example.ini` to `config/settings.ini`.
- Adjust settings as needed.

## Usage

Run the application using:

```bash
python src/main.py
```

## Configuration

### Settings

The application can be configured via the `config/settings.ini` file:

```ini
[Settings]
detection_threshold = 0.60
audio_threshold = 0.01
language = English
dark_mode = False
```

- **detection_threshold**: Confidence threshold for image detection (0.0 to 1.0).
- **audio_threshold**: Threshold for audio detection (float).
- **language**: Set to `English` or `French`. #can add more languages, just proof of concept
- **dark_mode**: Set to `True` or `False`.

### Adjusting Settings in the Application

You can also adjust settings from within the application by clicking on the **Settings** button.

## Directory Structure

```
drone-detection-system/
├── README.md
├── .gitignore
├── requirements.txt
├── config/
│   └── settings.ini
├── models/
│   └── best.pt
├── snapshots/
├── alert_history/
├── logs/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── detector.py
│   ├── gui.py
│   └── utils.py
└── tests/
    └── test_detector.py
```

- **config/**: Configuration files.
- **models/**: YOLO model weights.
- **snapshots/**: Saved images of detections.
- **alert_history/**: XML files of detection history.
- **logs/**: Log files for debugging.
- **src/**: Source code.
- **tests/**: Unit tests.

## Naming Conventions

- **Files and Directories**: Use lowercase letters and underscores (e.g., `detector.py`).
- **Classes**: Use CamelCase (e.g., `DroneDetector`).
- **Variables and Functions**: Use lowercase letters and underscores (e.g., `detection_threshold`).

## Backup and Version Control

- **Git**: Version control is managed using Git.
- **GitHub**: The repository is hosted on GitHub for collaboration and backup.
- **.gitignore**: Sensitive and unnecessary files are excluded via the `.gitignore` file.

