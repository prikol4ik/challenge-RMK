from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta, time
import polars as pl
import os
import logging

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
TALLINN_AGENCY_ID = "56"

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
if app.logger.handlers:
    for handler in app.logger.handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'))
else:
    app.logger.handlers = logging.getLogger().handlers

BUS_SCHEDULES_TD: list[tuple[timedelta, timedelta]] = []
MEETING_DATETIME_OBJ: datetime | None = None


def verify_gtfs_data_present(data_path: str) -> bool:
    if not os.path.isdir(data_path):
        app.logger.error(f"GTFS data directory '{data_path}' not found or is not a directory!")
        return False
    required_files = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt", "calendar.txt"]
    for req_file in required_files:
        if not os.path.exists(os.path.join(data_path, req_file)):
            app.logger.error(f"Required GTFS file '{req_file}' not found in '{data_path}'.")
            return False
    app.logger.info(f"GTFS data directory '{data_path}' found and all required files are present.")
    return True


def load_gtfs_table_polars(filename: str) -> pl.DataFrame | None:
    filepath = os.path.join(GTFS_DATA_DIR, filename)
    try:
        df = pl.read_csv(filepath, infer_schema_length=10000, null_values=["", "NA"])
        app.logger.debug(f"Successfully loaded {filepath} with Polars. Shape: {df.shape}")
        return df
    except Exception as e:
        app.logger.error(f"Error loading {filepath} with Polars: {e}", exc_info=True)
        return None


def parse_gtfs_time_to_polars_duration(df: pl.DataFrame, time_column_name: str) -> pl.Series:
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
    return (
            (temp_df["h_int"] * 3600 + temp_df["m_int"] * 60 + temp_df["s_int"]) * 1_000_000
    ).cast(pl.Duration(time_unit="us"))


def get_relevant_schedules_internal_polars() -> list[tuple[timedelta, timedelta]]:
    app.logger.info("Loading and processing GTFS schedule with Polars...")
    stops_df = load_gtfs_table_polars("stops.txt")
    routes_df = load_gtfs_table_polars("routes.txt")
    trips_df = load_gtfs_table_polars("trips.txt")
    stop_times_df = load_gtfs_table_polars("stop_times.txt")
    calendar_df = load_gtfs_table_polars("calendar.txt")

    if any(df is None for df in [stops_df, routes_df, trips_df, stop_times_df, calendar_df]):
        app.logger.error("One or more core GTFS tables were not loaded. Cannot proceed.")
        return []

    try:
        stops_df = stops_df.with_columns(pl.col("stop_id").cast(pl.Utf8))
        routes_df = routes_df.with_columns([
            pl.col("route_id").cast(pl.Utf8),
            pl.col("agency_id").cast(pl.Utf8),
            pl.col("route_short_name").cast(pl.Utf8),
            pl.col("route_type").cast(pl.Int32, strict=False)
        ])
        trips_df = trips_df.with_columns([
            pl.col("route_id").cast(pl.Utf8),
            pl.col("service_id").cast(pl.Utf8),
            pl.col("trip_id").cast(pl.Utf8)
        ])

        stop_times_df = stop_times_df.with_columns([
            pl.col("trip_id").cast(pl.Utf8),
            pl.col("stop_id").cast(pl.Utf8),
            pl.col("stop_sequence").cast(pl.Int32, strict=False)
        ])
        stop_times_df = stop_times_df.with_columns([
            parse_gtfs_time_to_polars_duration(stop_times_df, "departure_time").alias("departure_time_duration"),
            parse_gtfs_time_to_polars_duration(stop_times_df, "arrival_time").alias("arrival_time_duration")
        ])
        stop_times_df = stop_times_df.drop_nulls(subset=["departure_time_duration", "arrival_time_duration"])

        calendar_df = calendar_df.with_columns(pl.col("service_id").cast(pl.Utf8))

        zoo_stop_ids_series = stops_df.filter(pl.col("stop_name") == DEPARTURE_STOP_NAME).select("stop_id")["stop_id"]
        toompark_stop_ids_series = stops_df.filter(pl.col("stop_name") == ARRIVAL_STOP_NAME).select("stop_id")[
            "stop_id"]

        if zoo_stop_ids_series.is_empty() or toompark_stop_ids_series.is_empty():
            app.logger.error(f"Stop names '{DEPARTURE_STOP_NAME}' or '{ARRIVAL_STOP_NAME}' not found in stops.txt.")
            return []
        app.logger.info(
            f"Found {len(zoo_stop_ids_series)} stop_id(s) for '{DEPARTURE_STOP_NAME}': {zoo_stop_ids_series.to_list()}")
        app.logger.info(
            f"Found {len(toompark_stop_ids_series)} stop_id(s) for '{ARRIVAL_STOP_NAME}': {toompark_stop_ids_series.to_list()}")

        route_8_df = routes_df.filter(
            (pl.col("route_short_name") == BUS_ROUTE_SHORT_NAME) &
            (pl.col("route_type") == 3) &
            (pl.col("agency_id") == TALLINN_AGENCY_ID)
        )
        if route_8_df.is_empty():
            app.logger.error(f"Route '{BUS_ROUTE_SHORT_NAME}' (type 3, agency_id='{TALLINN_AGENCY_ID}') not found.")
            return []
        relevant_route_ids_series = route_8_df.select("route_id")["route_id"]
        app.logger.info(
            f"Using {len(relevant_route_ids_series)} route_id(s) for Tallinn bus {BUS_ROUTE_SHORT_NAME}: {relevant_route_ids_series.to_list()}")

    except Exception as e:
        app.logger.error(f"Error during ID retrieval or initial DataFrame processing: {e}", exc_info=True)
        return []

    weekday_service_ids_df = calendar_df.filter(
        (pl.col("monday") == 1) & (pl.col("tuesday") == 1) &
        (pl.col("wednesday") == 1) & (pl.col("thursday") == 1) &
        (pl.col("friday") == 1)
    ).select("service_id")
    app.logger.info(f"Weekday service IDs found: {weekday_service_ids_df.height}")
    if weekday_service_ids_df.is_empty():
        app.logger.warning("No weekday services found in calendar.txt.")
        return []

    relevant_trips_df = trips_df.filter(pl.col("route_id").is_in(relevant_route_ids_series)) \
        .join(weekday_service_ids_df, on="service_id", how="inner") \
        .select("trip_id")
    app.logger.info(f"Relevant trips (correct route, weekday): {relevant_trips_df.height} rows")
    if relevant_trips_df.is_empty():
        app.logger.warning("No suitable trips found for the route and days of the week.")
        return []

    stop_times_filtered_df = stop_times_df \
        .join(relevant_trips_df, on="trip_id", how="inner") \
        .filter(pl.col("stop_id").is_in(zoo_stop_ids_series) | pl.col("stop_id").is_in(toompark_stop_ids_series))

    app.logger.info(
        f"Stop times filtered (joined with trips, for Zoo/Toompark stops): {stop_times_filtered_df.height} rows")
    if stop_times_filtered_df.is_empty():
        app.logger.warning("No stop_times entries after filtering for relevant trips and stops.")
        return []
    if 0 < stop_times_filtered_df.height < 20:
        app.logger.debug(f"Sample of stop_times_filtered_df:\n{stop_times_filtered_df.head(20)}")

    zoo_departures = stop_times_filtered_df \
        .filter(pl.col("stop_id").is_in(zoo_stop_ids_series)) \
        .select(["trip_id", "departure_time_duration", "stop_sequence"]) \
        .rename({"departure_time_duration": "zoo_departure_td", "stop_sequence": "zoo_seq"})

    toompark_arrivals = stop_times_filtered_df \
        .filter(pl.col("stop_id").is_in(toompark_stop_ids_series)) \
        .select(["trip_id", "arrival_time_duration", "stop_sequence"]) \
        .rename({"arrival_time_duration": "toompark_arrival_td", "stop_sequence": "toompark_seq"})

    app.logger.info(f"Zoo departures records: {zoo_departures.height}")
    app.logger.info(f"Toompark arrivals records: {toompark_arrivals.height}")

    trip_full_info = zoo_departures.join(toompark_arrivals, on="trip_id", how="inner")
    app.logger.info(f"Trip full info (Zoo joined Toompark on trip_id): {trip_full_info.height} rows")
    if 0 < trip_full_info.height < 10:
        app.logger.debug(f"Sample of trip_full_info:\n{trip_full_info.head(10)}")
    if trip_full_info.is_empty():
        app.logger.warning("No trips found that serve both Zoo and Toompark with the given filters.")
        return []

    valid_trips_df = trip_full_info.filter(pl.col("zoo_seq") < pl.col("toompark_seq")) \
        .select(["zoo_departure_td", "toompark_arrival_td"]) \
        .sort("zoo_departure_td") \
        .unique(maintain_order=True)

    app.logger.info(f"Valid trips (Zoo seq < Toompark seq, sorted, unique times): {valid_trips_df.height} rows")

    schedules_list = []
    for row_tuple in valid_trips_df.iter_rows():
        dep_td: timedelta = row_tuple[0]
        arr_td: timedelta = row_tuple[1]
        schedules_list.append((dep_td, arr_td))

    app.logger.info(f"Found {len(schedules_list)} unique, valid trips from Zoo to Toompark to be used.")
    return schedules_list


def calculate_single_lateness(leave_home_time_str: str, bus_schedules_td: list[tuple[timedelta, timedelta]],
                              meeting_time_dt: datetime, walk_home_to_stop_sec: int,
                              walk_stop_to_meeting_sec: int) -> float:
    base_date = meeting_time_dt.date()
    try:
        if ':' in leave_home_time_str:
            parts = leave_home_time_str.split(':')
            if len(parts) == 2:
                leave_home_time_obj = datetime.strptime(leave_home_time_str, "%H:%M").time()
            elif len(parts) == 3:
                leave_home_time_obj = time.fromisoformat(leave_home_time_str)
            else:
                raise ValueError("Incorrect number of parts in time string.")
        else:
            raise ValueError("Time string must contain ':'.")
    except ValueError as e:
        app.logger.error(f"Error parsing time '{leave_home_time_str}': {e}")
        return -1.0

    leave_home_dt = datetime.combine(base_date, leave_home_time_obj)
    arrival_at_zoo_dt = leave_home_dt + timedelta(seconds=walk_home_to_stop_sec)
    app.logger.debug(f"Leaving home: {leave_home_dt}, Arriving at Zoo stop: {arrival_at_zoo_dt}")

    chosen_bus_arrival_at_toompark_dt = None
    for bus_dep_td, bus_arr_td in bus_schedules_td:
        bus_dep_zoo_dt = datetime.combine(base_date, time(0, 0, 0)) + bus_dep_td
        bus_arr_toompark_dt = datetime.combine(base_date, time(0, 0, 0)) + bus_arr_td
        if bus_dep_zoo_dt >= arrival_at_zoo_dt:
            chosen_bus_arrival_at_toompark_dt = bus_arr_toompark_dt
            app.logger.debug(f"Selected bus: Departs Zoo {bus_dep_zoo_dt}, Arrives Toompark {bus_arr_toompark_dt}")
            break

    if chosen_bus_arrival_at_toompark_dt is None:
        app.logger.warning(
            f"No suitable bus found after arriving at Zoo stop at {arrival_at_zoo_dt.time()}. Assuming late.")
        return 1.0

    final_arrival_at_meeting_dt = chosen_bus_arrival_at_toompark_dt + timedelta(seconds=walk_stop_to_meeting_sec)
    app.logger.debug(f"Final arrival at meeting room: {final_arrival_at_meeting_dt} (Meeting at {meeting_time_dt})")
    return 1.0 if final_arrival_at_meeting_dt > meeting_time_dt else 0.0


def generate_overview_plot_and_save(bus_schedules: list[tuple[timedelta, timedelta]],
                                    meeting_dt: datetime, plot_save_path: str) -> bool:
    if not bus_schedules:
        app.logger.warning("No schedule data to generate overview plot.")
        return False

    app.logger.info(f"Generating overview plot for meeting at {meeting_dt.strftime('%H:%M:%S')}...")
    base_date_for_plot = meeting_dt.date()
    start_calc_time_dt = datetime.combine(base_date_for_plot, time(6, 30, 0))
    end_calc_time_dt = datetime.combine(base_date_for_plot, time(9, 0, 1))

    departure_times_for_plot = []
    probabilities_for_plot = []
    current_time_dt = start_calc_time_dt
    while current_time_dt <= end_calc_time_dt:
        time_str = current_time_dt.strftime("%H:%M:%S")
        prob = calculate_single_lateness(
            time_str, bus_schedules, meeting_dt,
            WALK_HOME_TO_ZOO_SEC, WALK_TOOMPARK_TO_MEETING_SEC
        )
        if prob != -1.0:
            departure_times_for_plot.append(current_time_dt)
            probabilities_for_plot.append(prob)
        current_time_dt += timedelta(minutes=1)

    if not departure_times_for_plot:
        app.logger.warning("Failed to generate any data points for overview plot.")
        return False

    plt.figure(figsize=(12, 6))
    plt.plot(departure_times_for_plot, probabilities_for_plot, marker='.', linestyle='-', drawstyle='steps-post')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    plt.gcf().autofmt_xdate()
    plt.title(f"Rita's Lateness Probability (Meeting at {MEETING_TIME_STR})")
    plt.xlabel("Departure Time from Home")
    plt.ylabel("P(Late for Meeting)")
    plt.yticks([0, 1], ['On Time', 'Late'])
    plt.ylim(-0.1, 1.1)
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)
    plt.grid(True, which='major', axis='x', linestyle=':', linewidth=0.5)
    plt.tight_layout()

    try:
        static_dir_abs = os.path.abspath(os.path.dirname(plot_save_path))
        if not os.path.exists(static_dir_abs):
            os.makedirs(static_dir_abs, exist_ok=True)
            app.logger.info(f"Created static directory: {static_dir_abs}")
        plt.savefig(plot_save_path)
        app.logger.info(f"Overview plot saved to {plot_save_path}")
        plt.close()
        return True
    except Exception as e:
        app.logger.error(f"Failed to save overview plot: {e}", exc_info=True)
        plt.close()
        return False


def initialize_app_data():
    global BUS_SCHEDULES_TD, MEETING_DATETIME_OBJ
    app.logger.info("Initializing application data...")
    if not verify_gtfs_data_present(GTFS_DATA_DIR):
        app.logger.error("GTFS data verification failed. Bus schedules will be empty.")
        BUS_SCHEDULES_TD = []
    else:
        BUS_SCHEDULES_TD = get_relevant_schedules_internal_polars()

    if not BUS_SCHEDULES_TD:
        app.logger.warning("Bus schedule data is empty. Lateness calculations may not be accurate or possible.")
    else:
        app.logger.info(f"Successfully loaded {len(BUS_SCHEDULES_TD)} bus schedules into BUS_SCHEDULES_TD.")

    ref_date = datetime.today().date()
    MEETING_DATETIME_OBJ = datetime.combine(ref_date, time.fromisoformat(MEETING_TIME_STR))
    app.logger.info(f"Meeting datetime object set to: {MEETING_DATETIME_OBJ}")

    if BUS_SCHEDULES_TD:
        plot_generated = generate_overview_plot_and_save(BUS_SCHEDULES_TD, MEETING_DATETIME_OBJ, STATIC_PLOT_PATH)
        if not plot_generated:
            app.logger.warning("Overview plot generation failed or was skipped.")
    else:
        app.logger.warning("Overview plot will not be generated as there is no schedule data.")
    app.logger.info("Application data initialization complete.")


@app.route('/')
def index():
    plot_exists = os.path.exists(STATIC_PLOT_PATH)
    filename_for_template = PLOT_FILENAME if plot_exists else None
    cache_buster_value = datetime.now().timestamp() if plot_exists else None

    context = {
        "plot_image_filename": filename_for_template,
        "plot_cache_buster": cache_buster_value,
        "meeting_time_str": MEETING_TIME_STR,
        "walk_home_to_zoo_min": WALK_HOME_TO_ZOO_SEC / 60,
        "walk_toompark_to_meeting_min": WALK_TOOMPARK_TO_MEETING_SEC / 60,
        "bus_route": BUS_ROUTE_SHORT_NAME,
        "departure_stop": DEPARTURE_STOP_NAME,
        "arrival_stop": ARRIVAL_STOP_NAME,
        "schedules_loaded": bool(BUS_SCHEDULES_TD)
    }
    return render_template('index.html', **context)


@app.route('/calculate', methods=['POST'])
def calculate_api():
    data = request.get_json()
    if not data or 'leave_time' not in data:
        return jsonify({'error': 'Departure time (leave_time) not provided.'}), 400
    leave_time_str = data['leave_time']
    if not BUS_SCHEDULES_TD or MEETING_DATETIME_OBJ is None:
        app.logger.error("API /calculate: Bus schedules or meeting time not initialized.")
        return jsonify(
            {'error': 'Server error: Schedule data not loaded. Please check server logs or try again later.'}), 503
    probability = calculate_single_lateness(
        leave_time_str, BUS_SCHEDULES_TD, MEETING_DATETIME_OBJ,
        WALK_HOME_TO_ZOO_SEC, WALK_TOOMPARK_TO_MEETING_SEC
    )
    if probability == -1.0:
        return jsonify({'error': f"Invalid time format for '{leave_time_str}'. Use HH:MM or HH:MM:SS."}), 400
    is_late_bool = (probability == 1.0)
    status_message = "Rita will likely be LATE." if is_late_bool else "Rita will likely be ON TIME."
    return jsonify({
        'leave_time_processed': leave_time_str,
        'probability': probability,
        'is_late': is_late_bool,
        'status_message': status_message
    })


if __name__ == '__main__':
    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER, exist_ok=True)
        app.logger.info(f"Created static folder: {os.path.abspath(STATIC_FOLDER)}")

    if not os.path.exists("templates"):
        os.makedirs("templates", exist_ok=True)
        app.logger.info(f"Created templates folder: {os.path.abspath('templates')}. Ensure index.html is placed here.")

    initialize_app_data()
    app.logger.info("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5001)
else:
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        initialize_app_data()