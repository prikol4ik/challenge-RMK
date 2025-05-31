# Rita's Lateness Analyzer

This application analyzes the probability of Rita being late for her meeting using Tallinn's public transport GTFS (General Transit Feed Specification) data, her walking times, and potential bus delays.

This project was developed as a solution to the RMK Data Team Internship Challenge (2025).

## Problem Description

Rita works in the RMK Tallinn office. Every weekday (Mon-Fri), she has a meeting sharply at **09:05 AM**. To get to work, she takes city bus No. **8** from the **"Zoo"** stop to the **"Toompark"** stop.

*   The walk from home to the "Zoo" bus stop takes **300 seconds** (5 minutes).
*   The walk from the "Toompark" bus stop to the meeting room takes **240 seconds** (4 minutes).

The task is to visualize and calculate the probability of Rita being late depending on the time she leaves home, and also to account for possible bus delays.

## Tech Stack

This project is built using the following technologies:

*   **Backend & Core Logic:**
    *   **Python** (version 3.9+)
    *   **Flask:** Micro web framework for the web interface and API.
    *   **Polars:** High-performance DataFrame library for GTFS data processing.
    *   **Matplotlib:** Library for creating static, animated, and interactive visualizations (used for the lateness probability plot).
*   **Data Format:**
    *   **GTFS (General Transit Feed Specification):** Standardized format for public transportation schedules and associated geographic information.
*   **Frontend:**
    *   **HTML, CSS, JavaScript:** Standard web technologies for the user interface and interactivity.

## Functionality

*   **Lateness Probability Calculation**: Determines if Rita will make it to her meeting based on her departure time from home.
*   **Bus Delay Simulation**: Allows for two types of random delays to be considered:
    *   Maximum bus delay *before departure* from the "Zoo" stop.
    *   Maximum bus delay *after departure* from "Zoo" (on the way to "Toompark").
    *   The probability of lateness under these uncertain conditions is estimated using the **Monte Carlo method**, with `NUM_SIMULATIONS_FOR_DELAY` (default 1000) simulation runs.
*   **Overview Plot**: Displays Rita's lateness probability versus her departure time from home *under ideal conditions* (no bus delays).
*   **Interactive Calculator**: Allows the user to input their departure time from home and maximum anticipated bus delays to get a specific lateness probability.
*   **Schedule Display**: Shows the current weekday schedule for bus No. 8 on the specified route up to the meeting time.
*   **Web Interface**: Implemented using Flask.

## GTFS Data Source

The application uses data in the GTFS (General Transit Feed Specification) format.
*   **Specification for the Estonian National Public Transport Register (source for GTFS data):** [Ühistranspordiregister - Avaandmed spetsifikatsioon v1.4](https://www.agri.ee/sites/default/files/documents/2023-07/%C3%BChistranspordiregister-avaandmed-spetsifikatsioon-v1_4.pdf)

## Installation and Setup

### Prerequisites

*   Python 3.9+
*   pip (Python package manager)

### Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/prikol4ik/challenge-RMK
    cd challenge-RMK
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify GTFS data:**
    *   The necessary GTFS files are included in the `gtfs/` folder. Ensure this folder and its contents are present after cloning the repository.


5.  **Run the application:**
    ```bash
    python app.py
    ```

6.  **Open in your browser:**
    Navigate to [http://127.0.0.1:5001/](http://127.0.0.1:5001/).

## Assumptions and Limitations

*   **Schedule**: The application uses data for typical weekdays (Mon-Fri) active on the current date, based on the provided GTFS files.
*   **Bus Delays**: Modeled as random variables uniformly distributed between 0 and the specified maximum value for the Monte Carlo simulation.
*   **GTFS Accuracy**: Assumes the provided GTFS data is accurate for the period it represents.
*   **Route Choice**: Rita only uses bus No. 8.
*   **Walking Times**: Considered constant.
*   **Human Factor**: Rita boards the first suitable No. 8 bus.

