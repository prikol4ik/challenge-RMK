"""
Flask application to analyze Rita's probability of being late for a meeting.

This application uses GTFS (General Transit Feed Specification) data for Tallinn's
public transport to determine bus schedules. It then calculates the likelihood
of Rita being late for her 9:05 AM meeting based on her departure time from home,
walking times, and potential bus delays.

The application provides:
1. An overview plot showing lateness probability vs. departure time from home,
   assuming no bus delays.
2. An interactive calculator to find lateness probability for a specific
   departure time, considering user-defined maximum potential bus delays.
3. A display of relevant bus schedules.
"""
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta, time
import polars as pl
import os
import logging
import random
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

MEETING_TIME_STR = "09:05:00"
WALK_HOME_TO_ZOO_SEC = 300
WALK_TOOMPARK_TO_MEETING_SEC = 240
BUS_ROUTE_SHORT_NAME = "8"
DEPARTURE_STOP_NAME = "Zoo"
ARRIVAL_STOP_NAME = "Toompark"
GTFS_DATA_DIR = "gtfs"
PLOT_FILENAME = "rita_lateness_probability.png"
STATIC_FOLDER = "static"
STATIC_PLOT_PATH = os.path.join(STATIC_FOLDER, PLOT_FILENAME)
TALLINN_AGENCY_ID = "56"  # Agency ID for Tallinn's public transport in GTFS data (specific to this dataset)
NUM_SIMULATIONS_FOR_DELAY = 1000
DISPLAY_SCHEDULE_UNTIL_HOUR = 9
DISPLAY_SCHEDULE_UNTIL_MINUTE = 5
MAX_INPUT_DELAY_MINUTES = 15

app = Flask(__name__)
app.config['DISPLAY_SCHEDULE_UNTIL_HOUR'] = DISPLAY_SCHEDULE_UNTIL_HOUR
app.config['DISPLAY_SCHEDULE_UNTIL_MINUTE'] = DISPLAY_SCHEDULE_UNTIL_MINUTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

# Each tuple contains (departure_timedelta, arrival_timedelta) from midnight.
BUS_SCHEDULES_TD: list[tuple[timedelta, timedelta]] = []

# Used as the reference for lateness calculations.
MEETING_DATETIME_OBJ: datetime | None = None

# Each dictionary contains "departure_zoo" and "arrival_toompark" as "HH:MM" strings.
DISPLAYABLE_BUS_SCHEDULES: list[dict] = []


def load_gtfs_table_polars(filename: str) -> pl.DataFrame | None:
    """
    Load a GTFS data file (CSV) into a Polars DataFrame.

    Args:
        filename (str): The name of the GTFS file (e.g., "stops.txt").

    Returns:
        pl.DataFrame | None: A Polars DataFrame containing the data,
                             or None if an error occurs during loading.
    """
    filepath = os.path.join(GTFS_DATA_DIR, filename)
    try:
        df = pl.read_csv(filepath, infer_schema_length=10000, null_values=["", "NA"])
        return df
    except Exception as e:
        app.logger.error(f"Error loading {filepath}: {e}")
        return None


def parse_gtfs_time_to_polars_duration(df: pl.DataFrame, time_column_name: str) -> pl.Series:
    """
    Parse a GTFS time string column (e.g., "HH:MM:SS") into a Polars Duration series.

    GTFS times can exceed 24 hours (e.g., "25:00:00"), representing service continuing
    past midnight. This function correctly handles such times and converts them to
    timedeltas from the beginning of the service day.

    Args:
        df (pl.DataFrame): The Polars DataFrame containing the time column.
        time_column_name (str): The name of the column with GTFS time strings.

    Returns:
        pl.Series: A Polars Series of type Duration, representing the time as a timedelta.
                   Invalid time strings will result in nulls or default to 0 if parts are missing after parsing.
    """
    temp_df = df.with_columns(
        pl.col(time_column_name).str.split(":")
        .list.to_struct(n_field_strategy="first_non_null", fields=["h", "m", "s"])
        .alias("time_parts")
    )
    temp_df = temp_df.with_columns([
        pl.col("time_parts").struct.field("h").cast(pl.Int64, strict=False).fill_null(0).alias("h_int"),
        pl.col("time_parts").struct.field("m").cast(pl.Int64, strict=False).fill_null(0).alias("m_int"),
        pl.col("time_parts").struct.field("s").cast(pl.Int64, strict=False).fill_null(0).alias("s_int"),
    ])
    return ((temp_df["h_int"] * 3600 + temp_df["m_int"] * 60 + temp_df["s_int"]) * 1_000_000).cast(
        pl.Duration(time_unit="us"))


def get_relevant_schedules_internal_polars() -> list[tuple[timedelta, timedelta]]:
    """
    Process GTFS data to extract relevant bus schedules for Rita's commute.

    This function performs the following steps:
    1. Load necessary GTFS files (stops, routes, trips, stop_times, calendar).
    2. Cast relevant columns to appropriate Polars data types.
    3. Identify stop IDs for "Zoo" (departure) and "Toompark" (arrival).
    4. Filter routes to find bus route "8" operated by the specified Tallinn agency.
    5. Determine active service IDs for the current day (assuming typical Mon-Fri service).
       It prioritizes services active on all weekdays (Mon-Fri). If none are found,
       it falls back to services active on *any* weekday.
    6. Filter trips based on the selected route and active service IDs.
    7. Filter stop_times to include only those for relevant trips and stops.
    8. Join stop_times for "Zoo" departures with "Toompark" arrivals for the same trip.
    9. Validate that the "Zoo" stop sequence is before "Toompark" for each trip.
    10. Group by Zoo departure time and take the minimum Toompark arrival time.
    11. Sort schedules by departure time.
    12. Return a list of (departure_timedelta, arrival_timedelta) tuples.

    Returns:
        list[tuple[timedelta, timedelta]]: A list of schedules, where each tuple
        contains the departure time from Zoo and arrival time at Toompark,
        both as timedelta objects from midnight of the service day.
        Returns an empty list if data is missing or no relevant schedules are found.
    """
    stops_df = load_gtfs_table_polars("stops.txt")
    routes_df = load_gtfs_table_polars("routes.txt")
    trips_df = load_gtfs_table_polars("trips.txt")
    stop_times_df = load_gtfs_table_polars("stop_times.txt")
    calendar_df = load_gtfs_table_polars("calendar.txt")

    if any(df is None for df in [stops_df, routes_df, trips_df, stop_times_df, calendar_df]):
        app.logger.error("One or more GTFS files could not be loaded. Aborting schedule generation.")
        return []

    try:
        stops_df = stops_df.with_columns(pl.col("stop_id").cast(pl.Utf8))
        routes_df = routes_df.with_columns([
            pl.col("route_id").cast(pl.Utf8), pl.col("agency_id").cast(pl.Utf8),
            pl.col("route_short_name").cast(pl.Utf8), pl.col("route_type").cast(pl.Int32, strict=False)
        ])
        trips_df = trips_df.with_columns([
            pl.col("route_id").cast(pl.Utf8), pl.col("service_id").cast(pl.Utf8), pl.col("trip_id").cast(pl.Utf8)
        ])
        stop_times_df = stop_times_df.with_columns([
            pl.col("trip_id").cast(pl.Utf8), pl.col("stop_id").cast(pl.Utf8),
            pl.col("stop_sequence").cast(pl.Int32, strict=False),
            parse_gtfs_time_to_polars_duration(stop_times_df, "departure_time").alias("departure_time_duration"),
            parse_gtfs_time_to_polars_duration(stop_times_df, "arrival_time").alias("arrival_time_duration")
        ]).drop_nulls(subset=["departure_time_duration", "arrival_time_duration"])

        calendar_df = calendar_df.with_columns([
            pl.col("service_id").cast(pl.Utf8),
            pl.col("start_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("end_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
        ])

        # --- Filtering Step 1: Stops ---
        zoo_stop_ids = stops_df.filter(pl.col("stop_name") == DEPARTURE_STOP_NAME).select("stop_id")["stop_id"]
        toompark_stop_ids = stops_df.filter(pl.col("stop_name") == ARRIVAL_STOP_NAME).select("stop_id")["stop_id"]
        if zoo_stop_ids.is_empty() or toompark_stop_ids.is_empty():
            app.logger.error(f"Stop IDs not found for '{DEPARTURE_STOP_NAME}' or '{ARRIVAL_STOP_NAME}'.")
            return []

        # --- Filtering Step 2: Route ---
        route_8_ids = routes_df.filter(
            (pl.col("route_short_name") == BUS_ROUTE_SHORT_NAME) & \
            (pl.col("route_type") == 3) &
            (pl.col("agency_id") == TALLINN_AGENCY_ID)
        ).select("route_id")["route_id"]
        if route_8_ids.is_empty():
            app.logger.error(f"Route '{BUS_ROUTE_SHORT_NAME}' not found for agency '{TALLINN_AGENCY_ID}'.")
            return []

        # --- Filtering Step 3: Active Services (Calendar) ---
        current_date = datetime.now().date()
        # Use a fixed date for testing if needed:
        # current_date = datetime(2025, 6, 1).date()

        weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday"]
        if not all(col in calendar_df.columns for col in weekday_cols):
            app.logger.error(f"One or more weekday columns ({', '.join(weekday_cols)}) missing in calendar.txt.")
            return []

        # Find services active on all typical weekdays (Mon-Fri) and current date range
        active_weekday_services = calendar_df.filter(
            (pl.col("monday") == 1) & (pl.col("tuesday") == 1) & (pl.col("wednesday") == 1) &
            (pl.col("thursday") == 1) & (pl.col("friday") == 1) &
            (pl.col("start_date") <= current_date) & (pl.col("end_date") >= current_date)
        ).select("service_id")

        # Fallback: if no strictly Mon-Fri services, find services active on *any* weekday
        if active_weekday_services.is_empty():
            app.logger.warning(
                f"No strictly Mon-Fri services found for {current_date}. Looking for services active on any weekday.")
            active_weekday_services = calendar_df.filter(
                ((pl.col("monday") == 1) | (pl.col("tuesday") == 1) | (pl.col("wednesday") == 1) |
                 (pl.col("thursday") == 1) | (pl.col("friday") == 1)) &
                (pl.col("start_date") <= current_date) & (pl.col("end_date") >= current_date)
            ).select("service_id")

        if active_weekday_services.is_empty():
            app.logger.warning(f"No typical weekday services found active around {current_date.strftime('%Y-%m-%d')}.")
            return []
        # Get unique service IDs as a Polars Series
        active_service_ids = pl.Series("service_id", active_weekday_services["service_id"].unique().to_list(),
                                       dtype=pl.Utf8)

        # --- Filtering Step 4: Trips ---
        relevant_trips = trips_df.filter(
            pl.col("route_id").is_in(route_8_ids) & pl.col("service_id").is_in(active_service_ids)
        ).select("trip_id")
        if relevant_trips.is_empty():
            app.logger.info("No relevant trips found for the selected route and active services.")
            return []

        # --- Filtering Step 5: Stop Times ---
        stop_times_filtered = stop_times_df.join(relevant_trips, on="trip_id", how="inner").filter(
            pl.col("stop_id").is_in(zoo_stop_ids) | pl.col("stop_id").is_in(toompark_stop_ids)
        )
        if stop_times_filtered.is_empty():
            app.logger.info("No stop times found for the relevant trips at Zoo or Toompark.")
            return []

        # Separate Zoo departures and Toompark arrivals
        zoo_departures = stop_times_filtered.filter(pl.col("stop_id").is_in(zoo_stop_ids)).select(
            ["trip_id", "departure_time_duration", "stop_sequence"]
        ).rename({"departure_time_duration": "zoo_departure_td", "stop_sequence": "zoo_seq"})

        toompark_arrivals = stop_times_filtered.filter(pl.col("stop_id").is_in(toompark_stop_ids)).select(
            ["trip_id", "arrival_time_duration", "stop_sequence"]
        ).rename({"arrival_time_duration": "toompark_arrival_td", "stop_sequence": "toompark_seq"})

        # --- Joining and Finalizing Schedules ---
        trip_full_info = zoo_departures.join(toompark_arrivals, on="trip_id", how="inner")
        if trip_full_info.is_empty():
            app.logger.info("No trips found with both Zoo departure and Toompark arrival.")
            return []

        # Ensure Zoo departure occurs before Toompark arrival in the sequence
        valid_trips = trip_full_info.filter(pl.col("zoo_seq") < pl.col("toompark_seq")).select(
            ["zoo_departure_td", "toompark_arrival_td"]
        )
        if valid_trips.is_empty():
            app.logger.info("No valid trips found where Zoo stop sequence is before Toompark.")
            return []

        final_schedules_df = valid_trips.group_by("zoo_departure_td").agg(
            pl.min("toompark_arrival_td").alias("toompark_arrival_td")
        ).sort("zoo_departure_td")

        # Convert Polars Durations to Python timedelta objects
        schedules_list = [(row[0], row[1]) for row in final_schedules_df.iter_rows()]
        return schedules_list

    except Exception as e:
        app.logger.error(f"Unexpected error in get_relevant_schedules_internal_polars: {e}", exc_info=True)
        return []


def format_timedelta_to_time_str(td: timedelta) -> str:
    """
    Format a timedelta object into a "HH:MM" time string.

    Handles timedeltas that might represent times beyond 24 hours (e.g., from GTFS).

    Args:
        td (timedelta): The timedelta object to format.

    Returns:
        str: The formatted time string (e.g., "08:30", "25:15").
    """
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def calculate_single_lateness(
        leave_home_time_str: str,
        bus_schedules_td: list[tuple[timedelta, timedelta]],
        meeting_time_dt: datetime,
        walk_home_to_stop_sec: int,
        walk_stop_to_meeting_sec: int,
        max_bus_delay_at_zoo_minutes: int = 0,
        max_bus_delay_en_route_minutes: int = 0
) -> float:
    """
    Calculate the probability of Rita being late for her meeting.

    Args:
        leave_home_time_str (str): Rita's departure time from home (e.g., "08:10" or "08:10:00").
        bus_schedules_td (list[tuple[timedelta, timedelta]]): List of bus schedules
            (departure_timedelta, arrival_timedelta).
        meeting_time_dt (datetime): Datetime object of the meeting.
        walk_home_to_stop_sec (int): Walking time from home to bus stop in seconds.
        walk_stop_to_meeting_sec (int): Walking time from bus stop to meeting in seconds.
        max_bus_delay_at_zoo_minutes (int): Maximum potential delay of the bus
            AT the Zoo stop, in minutes. Delay is simulated uniformly between 0 and this value.
        max_bus_delay_en_route_minutes (int): Maximum potential additional travel time
            (delay EN ROUTE from Zoo to Toompark), in minutes. Simulated uniformly.

    Returns:
        float: Probability of being late (0.0 to 1.0).
               Returns -1.0 if `leave_home_time_str` is invalid.
               Returns 1.0 if no suitable bus is found.
    """
    base_date = meeting_time_dt.date()

    try:
        parts = leave_home_time_str.split(':')
        if len(parts) == 2:
            leave_home_time_obj = datetime.strptime(leave_home_time_str, "%H:%M").time()
        elif len(parts) == 3:
            leave_home_time_obj = time.fromisoformat(leave_home_time_str)
        else:
            raise ValueError("Incorrect time parts for leave_home_time_str")
    except ValueError:
        app.logger.error(f"Invalid time format for leave_home_time_str: '{leave_home_time_str}'")
        return -1.0

    rita_planned_arrival_at_zoo_dt = datetime.combine(base_date, leave_home_time_obj) + timedelta(
        seconds=walk_home_to_stop_sec)

    if not bus_schedules_td:
        return 1.0  # No buses, so definitely late

    # --- Scenario 1: No bus delays (deterministic calculation) ---
    if max_bus_delay_at_zoo_minutes == 0 and max_bus_delay_en_route_minutes == 0:
        chosen_bus_arrival_dt = None
        # Find the first bus Rita can catch
        for dep_td, arr_td in bus_schedules_td:
            # Convert schedule timedelta (from midnight of service day) to full datetime
            sched_dep_dt_zoo = datetime.combine(base_date, time.min) + dep_td
            if sched_dep_dt_zoo >= rita_planned_arrival_at_zoo_dt:
                chosen_bus_arrival_dt = datetime.combine(base_date, time.min) + arr_td
                break  # Rita takes this bus

        if chosen_bus_arrival_dt is None:
            return 1.0  # No bus found after Rita's arrival at Zoo, so she's late

        # Calculate final arrival time at the meeting
        final_arrival_at_meeting_dt = chosen_bus_arrival_dt + timedelta(seconds=walk_stop_to_meeting_sec)
        return 1.0 if final_arrival_at_meeting_dt > meeting_time_dt else 0.0

    # --- Scenario 2: Potential bus delays (probabilistic calculation using simulation) ---
    late_count = 0
    for _ in range(NUM_SIMULATIONS_FOR_DELAY):
        # For each simulation, re-determine the bus Rita aims for.
        # The critical part is how the *actual* departure/arrival of that bus is affected by random delays.

        # Find the first *scheduled* bus Rita can catch based on her planned arrival at Zoo
        chosen_bus_sim_info = None  # Stores (scheduled_departure_dt, scheduled_arrival_dt)
        for dep_td, arr_td in bus_schedules_td:
            sched_dep_dt_zoo = datetime.combine(base_date, time.min) + dep_td
            if sched_dep_dt_zoo >= rita_planned_arrival_at_zoo_dt:
                chosen_bus_sim_info = (sched_dep_dt_zoo, datetime.combine(base_date, time.min) + arr_td)
                break

        if chosen_bus_sim_info is None:
            late_count += 1  # No bus she can even aim for, definitely late for this simulation run
            continue

        _scheduled_dep_dt_zoo, scheduled_arrival_dt_toompark = chosen_bus_sim_info

        # Simulate delay at Zoo stop (bus arrives late or departs late from Zoo)
        delay_at_zoo_seconds = random.randint(0, max_bus_delay_at_zoo_minutes * 60)

        # Simulate additional delay en route (between Zoo and Toompark)
        delay_en_route_seconds = random.randint(0, max_bus_delay_en_route_minutes * 60)

        # Actual arrival time at Toompark includes both types of delays
        actual_arrival_at_toompark_dt = scheduled_arrival_dt_toompark + \
                                        timedelta(seconds=delay_at_zoo_seconds + delay_en_route_seconds)

        # Calculate final arrival time at the meeting for this simulation
        final_arrival_at_meeting_dt = actual_arrival_at_toompark_dt + timedelta(seconds=walk_stop_to_meeting_sec)

        if final_arrival_at_meeting_dt > meeting_time_dt:
            late_count += 1

    return late_count / NUM_SIMULATIONS_FOR_DELAY if NUM_SIMULATIONS_FOR_DELAY > 0 else 0.0


def generate_overview_plot_and_save(
        bus_schedules: list[tuple[timedelta, timedelta]],
        meeting_dt: datetime,
        plot_save_path: str
) -> bool:
    """
    Generate a plot of Rita's lateness probability vs. her departure time from home.

    This plot assumes NO bus delays (ideal conditions).

    Args:
        bus_schedules (list[tuple[timedelta, timedelta]]): Processed bus schedules.
        meeting_dt (datetime): Datetime of the meeting.
        plot_save_path (str): Full path where the plot image will be saved.

    Returns:
        bool: True if the plot was generated and saved successfully, False otherwise.
    """
    base_date = meeting_dt.date()

    # Define the time range for plotting departure times from home
    plot_end_dt = datetime.combine(base_date, meeting_dt.time()) - timedelta(minutes=5)  # Sensible upper limit
    plot_start_dt = max(datetime.combine(base_date, time(6, 0)), plot_end_dt - timedelta(hours=2, minutes=30))

    departure_times_for_plot = []
    probabilities_for_plot = []

    current_sim_dt = plot_start_dt
    while current_sim_dt <= plot_end_dt:
        # For this overview plot, calculate lateness assuming NO bus delays
        prob = calculate_single_lateness(
            current_sim_dt.strftime("%H:%M:%S"),
            bus_schedules,
            meeting_dt,
            WALK_HOME_TO_ZOO_SEC,
            WALK_TOOMPARK_TO_MEETING_SEC,
            max_bus_delay_at_zoo_minutes=0,
            max_bus_delay_en_route_minutes=0
        )
        if prob != -1.0:
            departure_times_for_plot.append(current_sim_dt)
            probabilities_for_plot.append(prob)
        current_sim_dt += timedelta(minutes=1)  # Increment by 1 minute for plot granularity

    if not departure_times_for_plot:
        app.logger.warning("No data points generated for the overview plot.")
        return False

    # --- Matplotlib Plotting ---
    plt.figure(figsize=(12, 6))
    plt.plot(departure_times_for_plot, probabilities_for_plot, marker='.', linestyle='-', drawstyle='steps-post')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))  # Ticks every 15 mins
    plt.gcf().autofmt_xdate()  # Auto-rotate date labels for readability

    plt.title(f"Rita's Lateness Probability (Meeting at {MEETING_TIME_STR}, Ideal Bus Conditions)")
    plt.xlabel("Departure Time from Home")
    plt.ylabel("P(Late for Meeting)")
    plt.yticks([0, 1], ['On Time', 'Late'])  # Probability is binary (0 or 1) for no-delay scenario
    plt.ylim(-0.1, 1.1)  # Add some padding to y-axis
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)
    plt.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5)
    plt.tight_layout()  # Adjust plot to ensure everything fits without overlapping

    try:
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)  # Ensure static directory exists
        plt.savefig(plot_save_path)
        app.logger.info(f"Overview plot saved to {plot_save_path}")
        return True
    except Exception as e:
        app.logger.error(f"Failed to save overview plot: {e}")
        return False
    finally:
        plt.close()  # Close the plot figure to free up memory


def initialize_app_data():
    """
    Initialize essential data for the application when it starts.

    This includes loading bus schedules, setting the meeting time,
    and generating the overview plot. This function populates the global variables:
    `BUS_SCHEDULES_TD`, `MEETING_DATETIME_OBJ`, `DISPLAYABLE_BUS_SCHEDULES`.
    """
    global BUS_SCHEDULES_TD, MEETING_DATETIME_OBJ, DISPLAYABLE_BUS_SCHEDULES
    app.logger.info("Initializing application data...")

    # Load and process GTFS data to get bus schedules
    BUS_SCHEDULES_TD = get_relevant_schedules_internal_polars()
    if not BUS_SCHEDULES_TD:
        app.logger.warning("Bus schedule data is empty after processing. App may not function as expected.")

    DISPLAYABLE_BUS_SCHEDULES = []
    if BUS_SCHEDULES_TD:
        cutoff_time = time(app.config['DISPLAY_SCHEDULE_UNTIL_HOUR'], app.config['DISPLAY_SCHEDULE_UNTIL_MINUTE'])
        cutoff_td = timedelta(hours=cutoff_time.hour, minutes=cutoff_time.minute)
        for dep_td, arr_td in BUS_SCHEDULES_TD:
            if dep_td <= cutoff_td:
                DISPLAYABLE_BUS_SCHEDULES.append({
                    "departure_zoo": format_timedelta_to_time_str(dep_td),
                    "arrival_toompark": format_timedelta_to_time_str(arr_td)
                })

    # Set the meeting datetime object for today
    app_load_date = datetime.now().date()
    try:
        meeting_time_obj = time.fromisoformat(MEETING_TIME_STR)
        MEETING_DATETIME_OBJ = datetime.combine(app_load_date, meeting_time_obj)
    except ValueError:
        app.logger.error(f"Invalid MEETING_TIME_STR: {MEETING_TIME_STR}. Meeting time could not be set.")
        MEETING_DATETIME_OBJ = None

    # Generate and save the overview plot if data is available
    if BUS_SCHEDULES_TD and MEETING_DATETIME_OBJ:
        if not generate_overview_plot_and_save(BUS_SCHEDULES_TD, MEETING_DATETIME_OBJ, STATIC_PLOT_PATH):
            app.logger.warning("Overview plot generation failed. Check previous logs.")
    else:
        app.logger.warning("Overview plot not generated due to missing bus schedule data or meeting time.")

    app.logger.info("Application data initialization complete.")


@app.route('/')
def index():
    """
    Serve the main HTML page of the application.

    Provide context data to the template, including plot information,
    fixed parameters (walking times, meeting time), and bus schedules.
    """
    plot_exists = os.path.exists(STATIC_PLOT_PATH)
    context = {
        "plot_image_filename": PLOT_FILENAME if plot_exists else None,
        "plot_cache_buster": datetime.now().timestamp() if plot_exists else None,
        "meeting_time_str": MEETING_DATETIME_OBJ.strftime("%H:%M:%S") if MEETING_DATETIME_OBJ else MEETING_TIME_STR,
        "walk_home_to_zoo_min": WALK_HOME_TO_ZOO_SEC / 60,
        "walk_toompark_to_meeting_min": WALK_TOOMPARK_TO_MEETING_SEC / 60,
        "bus_route": BUS_ROUTE_SHORT_NAME,
        "departure_stop": DEPARTURE_STOP_NAME,
        "arrival_stop": ARRIVAL_STOP_NAME,
        "schedules_loaded": bool(BUS_SCHEDULES_TD),
        "bus_schedules_for_display": DISPLAYABLE_BUS_SCHEDULES,
    }
    return render_template('index.html', **context)


@app.route('/calculate', methods=['POST'])
def calculate_api():
    """
    Handle API requests to calculate lateness probability.

    Accept a JSON POST request with 'leave_time', 'max_bus_delay_at_zoo_minutes',
    and 'max_bus_delay_en_route_minutes'.
    Return a JSON response with the calculated probability and a status message.
    """
    data = request.get_json()
    if not data or 'leave_time' not in data:
        return jsonify({'error': 'Departure time ("leave_time") not provided.'}), 400

    leave_time_str = data['leave_time']

    try:
        delay_zoo_min = max(0, min(int(data.get('max_bus_delay_at_zoo_minutes', 0)), MAX_INPUT_DELAY_MINUTES))
        delay_route_min = max(0, min(int(data.get('max_bus_delay_en_route_minutes', 0)), MAX_INPUT_DELAY_MINUTES))
    except (ValueError, TypeError):
        app.logger.warning(
            f"Invalid delay input received: {data.get('max_bus_delay_at_zoo_minutes')}, {data.get('max_bus_delay_en_route_minutes')}. Using defaults (0).")
        delay_zoo_min, delay_route_min = 0, 0

    if MEETING_DATETIME_OBJ is None:
        app.logger.error("Meeting datetime object is not initialized. Cannot perform calculation.")
        return jsonify({
                           'error': 'Server configuration error: Meeting time not set. Please check server logs.'}), 503  # Service Unavailable
    if not BUS_SCHEDULES_TD:
        app.logger.error("Bus schedules are not loaded. Cannot perform calculation.")
        return jsonify({'error': 'Server error: Bus schedule data unavailable. Please check server logs.'}), 503

    probability = calculate_single_lateness(
        leave_time_str, BUS_SCHEDULES_TD, MEETING_DATETIME_OBJ,
        WALK_HOME_TO_ZOO_SEC, WALK_TOOMPARK_TO_MEETING_SEC,
        max_bus_delay_at_zoo_minutes=delay_zoo_min,
        max_bus_delay_en_route_minutes=delay_route_min
    )

    if probability == -1.0:
        return jsonify(
            {'error': f"Invalid time format for 'leave_time': '{leave_time_str}'. Please use HH:MM or HH:MM:SS."}), 400

    prob_percent = probability * 100
    status_msg = ""
    if delay_zoo_min == 0 and delay_route_min == 0:  # Deterministic case
        status_msg = "Rita will be LATE." if probability == 1.0 else "Rita will be ON TIME."
    else:
        if probability >= 0.99:
            status_msg = f"Rita will almost certainly be LATE (P(Late) = {prob_percent:.0f}%)."
        elif probability >= 0.75:
            status_msg = f"Rita is LIKELY to be LATE (P(Late) = {prob_percent:.0f}%)."
        elif probability >= 0.25:
            status_msg = f"Rita MIGHT be late (P(Late) = {prob_percent:.0f}%)."
        elif probability > 0.001:
            status_msg = f"Rita is very likely ON TIME, small chance of being late (P(Late) = {prob_percent:.1f}%)."
        else:
            status_msg = f"Rita will almost certainly be ON TIME (P(Late) = {prob_percent:.0f}%)."

    return jsonify({
        'leave_time_processed': leave_time_str,
        'probability': probability,
        'is_late': (probability > 0.5),
        'status_message': status_msg
    })


if __name__ == '__main__':
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    initialize_app_data()
    app.logger.info("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5001)