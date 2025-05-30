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
TALLINN_AGENCY_ID = "56"
NUM_SIMULATIONS_FOR_DELAY = 1000
DISPLAY_SCHEDULE_UNTIL_HOUR = 9
DISPLAY_SCHEDULE_UNTIL_MINUTE = 5
MAX_INPUT_DELAY_MINUTES = 15

app = Flask(__name__)
app.config['DISPLAY_SCHEDULE_UNTIL_HOUR'] = DISPLAY_SCHEDULE_UNTIL_HOUR
app.config['DISPLAY_SCHEDULE_UNTIL_MINUTE'] = DISPLAY_SCHEDULE_UNTIL_MINUTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

BUS_SCHEDULES_TD: list[tuple[timedelta, timedelta]] = []
MEETING_DATETIME_OBJ: datetime | None = None
DISPLAYABLE_BUS_SCHEDULES: list[dict] = []


def verify_gtfs_data_present(data_path: str) -> bool:
    if not os.path.isdir(data_path):
        app.logger.error(f"GTFS data directory '{data_path}' not found or is not a directory!")
        return False
    required_files = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt", "calendar.txt"]
    for req_file in required_files:
        if not os.path.exists(os.path.join(data_path, req_file)):
            app.logger.error(f"Required GTFS file '{req_file}' not found in '{data_path}'.")
            return False
    if not os.path.exists(os.path.join(data_path, "calendar_dates.txt")):
        app.logger.warning(f"Optional GTFS file 'calendar_dates.txt' not found in '{data_path}'. "
                           f"Calculations will proceed but may not reflect specific date exceptions.")
    app.logger.info(
        f"GTFS data directory '{data_path}' found and all required files are present (calendar_dates.txt is optional).")
    return True


def load_gtfs_table_polars(filename: str) -> pl.DataFrame | None:
    filepath = os.path.join(GTFS_DATA_DIR, filename)
    try:
        df = pl.read_csv(filepath, infer_schema_length=10000, null_values=["", "NA"])
        app.logger.debug(f"Successfully loaded {filepath} with Polars. Shape: {df.shape}")
        return df
    except Exception as e:
        if isinstance(e, FileNotFoundError) and filename == "calendar_dates.txt":
            app.logger.warning(f"Optional file {filepath} not found. Proceeding without it.")
            return None
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
    app.logger.info("Loading and processing GTFS schedule for a TYPICAL WORKDAY (Mon-Fri), "
                    "ignoring specific date exceptions like holidays, but respecting service date ranges.")
    stops_df = load_gtfs_table_polars("stops.txt")
    routes_df = load_gtfs_table_polars("routes.txt")
    trips_df = load_gtfs_table_polars("trips.txt")
    stop_times_df = load_gtfs_table_polars("stop_times.txt")
    calendar_df = load_gtfs_table_polars("calendar.txt")


    if any(df is None for df in [stops_df, routes_df, trips_df, stop_times_df, calendar_df]):
        app.logger.error("One or more core GTFS files could not be loaded. Aborting schedule generation.")
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

        calendar_df = calendar_df.with_columns([
            pl.col("service_id").cast(pl.Utf8),
            pl.col("start_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False),
            pl.col("end_date").cast(pl.Utf8).str.to_date("%Y%m%d", strict=False)
        ])

        zoo_stop_ids_series = stops_df.filter(pl.col("stop_name") == DEPARTURE_STOP_NAME).select("stop_id")["stop_id"]
        toompark_stop_ids_series = stops_df.filter(pl.col("stop_name") == ARRIVAL_STOP_NAME).select("stop_id")[
            "stop_id"]

        if zoo_stop_ids_series.is_empty() or toompark_stop_ids_series.is_empty():
            app.logger.error(f"Could not find stop IDs for '{DEPARTURE_STOP_NAME}' or '{ARRIVAL_STOP_NAME}'.")
            return []

        route_8_df = routes_df.filter(
            (pl.col("route_short_name") == BUS_ROUTE_SHORT_NAME) &
            (pl.col("route_type") == 3) &
            (pl.col("agency_id") == TALLINN_AGENCY_ID)
        )
        if route_8_df.is_empty():
            app.logger.error(f"Could not find route '{BUS_ROUTE_SHORT_NAME}' for agency '{TALLINN_AGENCY_ID}'.")
            return []
        relevant_route_ids_series = route_8_df.select("route_id")["route_id"]

    except Exception as e:
        app.logger.error(f"Error during ID retrieval or initial DataFrame processing: {e}", exc_info=True)
        return []


    current_date = datetime.now().date()
    app.logger.info(f"Filtering services active in the period covering {current_date.strftime('%Y-%m-%d')} "
                    f"and that operate on a typical Mon-Fri schedule.")

    weekday_cols = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    for col_name in weekday_cols:
        if col_name not in calendar_df.columns:
            app.logger.error(
                f"Required weekday column '{col_name}' not found in calendar.txt. Available columns: {calendar_df.columns}")
            return []

    active_weekday_services = calendar_df.filter(
        (pl.col("monday") == 1) &
        (pl.col("tuesday") == 1) &
        (pl.col("wednesday") == 1) &
        (pl.col("thursday") == 1) &
        (pl.col("friday") == 1) &
        (pl.col("start_date") <= current_date) &
        (pl.col("end_date") >= current_date)
    ).select("service_id")
    app.logger.debug(f"Found {active_weekday_services.height} services active Mon-Fri AND within current date range.")


    if active_weekday_services.is_empty():
        app.logger.info("No services found active on ALL Mon-Fri. Trying services active on ANY Mon-Fri.")
        active_weekday_services = calendar_df.filter(
            (
                    (pl.col("monday") == 1) |
                    (pl.col("tuesday") == 1) |
                    (pl.col("wednesday") == 1) |
                    (pl.col("thursday") == 1) |
                    (pl.col("friday") == 1)
            ) &
            (pl.col("start_date") <= current_date) &
            (pl.col("end_date") >= current_date)
        ).select("service_id")
        app.logger.debug(
            f"Found {active_weekday_services.height} services active on ANY Mon-Fri AND within current date range.")

    if active_weekday_services.is_empty():
        app.logger.warning(
            f"No typical Mon-Fri services found that are active around {current_date.strftime('%Y-%m-%d')}.")
        return []

    active_service_ids_final_list = list(active_weekday_services["service_id"].unique().to_list())

    app.logger.info(
        f"Found {len(active_service_ids_final_list)} typical Mon-Fri service_id(s) active in the current period: "
        f"{active_service_ids_final_list[:20]}{'...' if len(active_service_ids_final_list) > 20 else ''}")
    active_service_ids_pl_series = pl.Series("service_id", active_service_ids_final_list, dtype=pl.Utf8)

    relevant_trips_df = trips_df.filter(
        pl.col("route_id").is_in(relevant_route_ids_series) &
        pl.col("service_id").is_in(active_service_ids_pl_series)
    ).select("trip_id")

    if relevant_trips_df.is_empty():
        app.logger.warning(
            f"No relevant trips found for route '{BUS_ROUTE_SHORT_NAME}' using typical Mon-Fri active services.")
        return []

    stop_times_filtered_df = stop_times_df \
        .join(relevant_trips_df, on="trip_id", how="inner") \
        .filter(pl.col("stop_id").is_in(zoo_stop_ids_series) | pl.col("stop_id").is_in(toompark_stop_ids_series))

    if stop_times_filtered_df.is_empty():
        app.logger.warning(
            f"No stop times found for relevant trips at '{DEPARTURE_STOP_NAME}' or '{ARRIVAL_STOP_NAME}'.")
        return []

    zoo_departures = stop_times_filtered_df \
        .filter(pl.col("stop_id").is_in(zoo_stop_ids_series)) \
        .select(["trip_id", "departure_time_duration", "stop_sequence"]) \
        .rename({"departure_time_duration": "zoo_departure_td", "stop_sequence": "zoo_seq"})

    toompark_arrivals = stop_times_filtered_df \
        .filter(pl.col("stop_id").is_in(toompark_stop_ids_series)) \
        .select(["trip_id", "arrival_time_duration", "stop_sequence"]) \
        .rename({"arrival_time_duration": "toompark_arrival_td", "stop_sequence": "toompark_seq"})

    trip_full_info = zoo_departures.join(toompark_arrivals, on="trip_id", how="inner")

    if trip_full_info.is_empty():
        app.logger.warning("No trips found with both departure from Zoo and arrival at Toompark.")
        return []

    valid_trips_intermediate_df = trip_full_info.filter(pl.col("zoo_seq") < pl.col("toompark_seq")) \
        .select(["zoo_departure_td", "toompark_arrival_td"])

    if valid_trips_intermediate_df.is_empty():
        app.logger.warning("No valid trips found where Zoo stop sequence is before Toompark stop sequence.")
        return []

    final_valid_trips_df = valid_trips_intermediate_df.group_by("zoo_departure_td").agg(
        pl.min("toompark_arrival_td").alias("earliest_toompark_arrival_td")
    ).sort("zoo_departure_td")

    final_valid_trips_df = final_valid_trips_df.rename({"earliest_toompark_arrival_td": "toompark_arrival_td"})

    schedules_list = [(row[0], row[1]) for row in final_valid_trips_df.iter_rows()]
    app.logger.info(f"Successfully generated {len(schedules_list)} unique schedules for a typical Mon-Fri workday "
                    f"from '{DEPARTURE_STOP_NAME}' to '{ARRIVAL_STOP_NAME}'.")
    return schedules_list


def format_timedelta_to_time_str(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    return f"{hours:02d}:{minutes:02d}"


def calculate_single_lateness(leave_home_time_str: str,
                              bus_schedules_td: list[tuple[timedelta, timedelta]],
                              meeting_time_dt: datetime,
                              walk_home_to_stop_sec: int,
                              walk_stop_to_meeting_sec: int,
                              max_bus_delay_at_zoo_minutes: int = 0,
                              max_bus_delay_en_route_minutes: int = 0
                              ) -> float:

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

    rita_planned_arrival_at_zoo_dt = datetime.combine(base_date, leave_home_time_obj) + timedelta(
        seconds=walk_home_to_stop_sec)

    app.logger.debug(
        f"--- Lateness Calculation Start ---\n"
        f"Leave Home Time: {leave_home_time_obj.strftime('%H:%M:%S')}\n"
        f"Rita Planned Arrival at Zoo (on {base_date.isoformat()}): {rita_planned_arrival_at_zoo_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Meeting Time (on {base_date.isoformat()}): {meeting_time_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Max Delay at Zoo: {max_bus_delay_at_zoo_minutes} min\n"
        f"Max Delay En Route: {max_bus_delay_en_route_minutes} min"
    )

    if not bus_schedules_td:
        app.logger.warning("No bus schedules provided for lateness calculation. Assuming Rita will be late.")
        return 1.0

    if max_bus_delay_at_zoo_minutes == 0 and max_bus_delay_en_route_minutes == 0:
        app.logger.debug("Calculating deterministically (no bus delays).")

        for bus_dep_td, bus_arr_td in bus_schedules_td:

            current_scheduled_bus_dep_dt = datetime.combine(base_date, time.min) + bus_dep_td



            if current_scheduled_bus_dep_dt >= rita_planned_arrival_at_zoo_dt:
                chosen_bus_scheduled_arrival_dt_at_toompark = datetime.combine(base_date, time.min) + bus_arr_td
                app.logger.debug(
                    f"Deterministic: Found bus. Scheduled Dep Zoo: {current_scheduled_bus_dep_dt.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"Scheduled Arr Toompark: {chosen_bus_scheduled_arrival_dt_at_toompark.strftime('%Y-%m-%d %H:%M:%S')}")
                break

        if chosen_bus_scheduled_arrival_dt_at_toompark is None:
            app.logger.debug("Deterministic: No suitable bus found for Rita.")
            return 1.0

        final_arrival_at_meeting_dt = chosen_bus_scheduled_arrival_dt_at_toompark + timedelta(
            seconds=walk_stop_to_meeting_sec)
        is_late = final_arrival_at_meeting_dt > meeting_time_dt
        app.logger.debug(
            f"Deterministic: Final arrival at meeting: {final_arrival_at_meeting_dt.strftime('%Y-%m-%d %H:%M:%S')}. Late: {is_late}")
        return 1.0 if is_late else 0.0

    app.logger.debug(f"Starting simulation with {NUM_SIMULATIONS_FOR_DELAY} iterations.")
    late_count = 0
    for i in range(NUM_SIMULATIONS_FOR_DELAY):
        chosen_bus_info_for_sim = None

        for bus_dep_td, bus_arr_td in bus_schedules_td:
            scheduled_dep_dt_candidate = datetime.combine(base_date, time.min) + bus_dep_td
            if scheduled_dep_dt_candidate >= rita_planned_arrival_at_zoo_dt:
                scheduled_arr_dt_candidate = datetime.combine(base_date, time.min) + bus_arr_td
                chosen_bus_info_for_sim = (scheduled_dep_dt_candidate, scheduled_arr_dt_candidate)
                if i == 0:
                    app.logger.debug(
                        f"Sim 0: Rita aims for bus. Sched Dep Zoo: {scheduled_dep_dt_candidate.strftime('%H:%M:%S')}, "
                        f"Sched Arr Toompark: {scheduled_arr_dt_candidate.strftime('%H:%M:%S')}")
                break

        if chosen_bus_info_for_sim is None:
            late_count += 1
            if i == 0: app.logger.debug("Sim 0: No scheduled bus for Rita to aim for. Counted as late.")
            continue

        scheduled_dep_dt_for_sim_bus_at_zoo, scheduled_arr_dt_for_sim_bus_at_toompark = chosen_bus_info_for_sim

        delay_at_zoo_sec = random.randint(0, max_bus_delay_at_zoo_minutes * 60)
        actual_bus_departure_from_zoo_dt = scheduled_dep_dt_for_sim_bus_at_zoo + timedelta(seconds=delay_at_zoo_sec)

        delay_en_route_sec = random.randint(0, max_bus_delay_en_route_minutes * 60)

        actual_bus_arrival_at_toompark_dt = scheduled_arr_dt_for_sim_bus_at_toompark + \
                                            timedelta(seconds=delay_at_zoo_sec) + \
                                            timedelta(seconds=delay_en_route_sec)

        final_arrival_at_meeting_dt = actual_bus_arrival_at_toompark_dt + timedelta(seconds=walk_stop_to_meeting_sec)

        if final_arrival_at_meeting_dt > meeting_time_dt:
            late_count += 1
            if i < 3 and late_count <= 3:
                app.logger.debug(
                    f"Sim {i}: LATE \n"
                    f"  Rita Planned Arr Zoo: {rita_planned_arrival_at_zoo_dt.strftime('%H:%M:%S')}\n"
                    f"  Bus Sched Dep Zoo: {scheduled_dep_dt_for_sim_bus_at_zoo.strftime('%H:%M:%S')}, Sched Arr Toompark: {scheduled_arr_dt_for_sim_bus_at_toompark.strftime('%H:%M:%S')}\n"
                    f"  Delay at Zoo: {delay_at_zoo_sec}s -> Actual Dep Zoo: {actual_bus_departure_from_zoo_dt.strftime('%H:%M:%S')}\n"
                    f"  Delay En Route: {delay_en_route_sec}s\n"
                    f"  Actual Arr Toompark: {actual_bus_arrival_at_toompark_dt.strftime('%H:%M:%S')}\n"
                    f"  Final Arr Meeting: {final_arrival_at_meeting_dt.strftime('%H:%M:%S')} (vs {meeting_time_dt.strftime('%H:%M:%S')})"
                )

    probability_of_lateness = late_count / NUM_SIMULATIONS_FOR_DELAY if NUM_SIMULATIONS_FOR_DELAY > 0 else 0.0
    app.logger.debug(
        f"Simulation ended. Late count: {late_count}/{NUM_SIMULATIONS_FOR_DELAY}. Probability: {probability_of_lateness:.4f}\n"
        f"--- Lateness Calculation End ---"
    )
    return probability_of_lateness


def generate_overview_plot_and_save(bus_schedules: list[tuple[timedelta, timedelta]],
                                    meeting_dt: datetime, plot_save_path: str) -> bool:
    if not bus_schedules:
        app.logger.warning("No bus schedules to generate plot.")
        return False
    app.logger.info(
        f"Generating overview plot for meeting at {meeting_dt.strftime('%H:%M:%S')} (ideal bus conditions)...")

    base_date_for_plot = meeting_dt.date()



    meeting_time_only = meeting_dt.time()
    plot_end_dt = datetime.combine(base_date_for_plot, meeting_time_only) - timedelta(minutes=5)
    plot_start_dt = max(
        datetime.combine(base_date_for_plot, time(6, 0, 0)),
        plot_end_dt - timedelta(hours=2, minutes=30)
    )

    app.logger.debug(f"Plot generation range: {plot_start_dt.strftime('%H:%M')} to {plot_end_dt.strftime('%H:%M')}")

    departure_times_for_plot = []
    probabilities_for_plot = []
    current_time_dt = plot_start_dt

    while current_time_dt <= plot_end_dt:
        time_str = current_time_dt.strftime("%H:%M:%S")
        prob = calculate_single_lateness(
            time_str, bus_schedules, meeting_dt,
            WALK_HOME_TO_ZOO_SEC, WALK_TOOMPARK_TO_MEETING_SEC,
            max_bus_delay_at_zoo_minutes=0,
            max_bus_delay_en_route_minutes=0
        )
        if prob != -1.0:
            departure_times_for_plot.append(current_time_dt)
            probabilities_for_plot.append(prob)
        current_time_dt += timedelta(minutes=1)

    if not departure_times_for_plot:
        app.logger.warning("No data points generated for the overview plot.")
        return False

    plt.figure(figsize=(12, 6))
    plt.plot(departure_times_for_plot, probabilities_for_plot, marker='.', linestyle='-', drawstyle='steps-post')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 15)))
    plt.gcf().autofmt_xdate()

    plt.title(f"Rita's Lateness Probability (Meeting at {MEETING_TIME_STR}, Ideal Bus Conditions)")
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
        plt.close()
        app.logger.info(f"Overview plot saved to {plot_save_path}")
        return True
    except Exception as e:
        app.logger.error(f"Failed to save overview plot: {e}", exc_info=True)
        plt.close()
        return False


def initialize_app_data():
    global BUS_SCHEDULES_TD, MEETING_DATETIME_OBJ, DISPLAYABLE_BUS_SCHEDULES
    app.logger.info("Initializing application data...")

    if not verify_gtfs_data_present(GTFS_DATA_DIR):
        BUS_SCHEDULES_TD = []
        app.logger.error("GTFS data verification failed. Application will run with no schedule data.")
    else:
        BUS_SCHEDULES_TD = get_relevant_schedules_internal_polars()

    if not BUS_SCHEDULES_TD:
        app.logger.warning("Bus schedule data is empty after processing. Check GTFS data and filters.")
    else:
        app.logger.info(f"Successfully loaded {len(BUS_SCHEDULES_TD)} currently active bus schedules.")

        DISPLAYABLE_BUS_SCHEDULES = []
        cutoff_time_obj = time(app.config['DISPLAY_SCHEDULE_UNTIL_HOUR'], app.config['DISPLAY_SCHEDULE_UNTIL_MINUTE'],
                               0)
        cutoff_timedelta = timedelta(hours=cutoff_time_obj.hour, minutes=cutoff_time_obj.minute,
                                     seconds=cutoff_time_obj.second)

        for dep_td, arr_td in BUS_SCHEDULES_TD:
            if dep_td <= cutoff_timedelta:
                DISPLAYABLE_BUS_SCHEDULES.append({
                    "departure_zoo": format_timedelta_to_time_str(dep_td),
                    "arrival_toompark": format_timedelta_to_time_str(arr_td)
                })
        app.logger.info(
            f"Prepared {len(DISPLAYABLE_BUS_SCHEDULES)} schedules for display (until {cutoff_time_obj.strftime('%H:%M')})."
        )

    app_load_date = datetime.now().date()
    try:
        meeting_time_parsed = time.fromisoformat(MEETING_TIME_STR)
        MEETING_DATETIME_OBJ = datetime.combine(app_load_date, meeting_time_parsed)
        app.logger.info(f"Meeting datetime set to: {MEETING_DATETIME_OBJ.strftime('%Y-%m-%d %H:%M:%S')}")
    except ValueError:
        app.logger.error(f"Invalid MEETING_TIME_STR: {MEETING_TIME_STR}. Cannot set meeting datetime.")
        MEETING_DATETIME_OBJ = None

    if BUS_SCHEDULES_TD and MEETING_DATETIME_OBJ:
        plot_generated = generate_overview_plot_and_save(BUS_SCHEDULES_TD, MEETING_DATETIME_OBJ, STATIC_PLOT_PATH)
        if not plot_generated:
            app.logger.warning("Overview plot generation failed.")
    elif not MEETING_DATETIME_OBJ:
        app.logger.warning("Overview plot not generated (meeting datetime not set).")
    else:
        app.logger.warning("Overview plot not generated (no schedule data).")

    app.logger.info("Application data initialization complete.")


@app.route('/')
def index():
    plot_exists = os.path.exists(STATIC_PLOT_PATH)
    filename_for_template = PLOT_FILENAME if plot_exists else None
    cache_buster_value = datetime.now().timestamp() if plot_exists else None

    meeting_time_display = MEETING_TIME_STR
    if MEETING_DATETIME_OBJ:
        meeting_time_display = MEETING_DATETIME_OBJ.strftime("%H:%M:%S")

    context = {
        "plot_image_filename": filename_for_template,
        "plot_cache_buster": cache_buster_value,
        "meeting_time_str": meeting_time_display,
        "walk_home_to_zoo_min": WALK_HOME_TO_ZOO_SEC / 60,
        "walk_toompark_to_meeting_min": WALK_TOOMPARK_TO_MEETING_SEC / 60,
        "bus_route": BUS_ROUTE_SHORT_NAME,
        "departure_stop": DEPARTURE_STOP_NAME,
        "arrival_stop": ARRIVAL_STOP_NAME,
        "schedules_loaded": bool(BUS_SCHEDULES_TD),
        "bus_schedules_for_display": DISPLAYABLE_BUS_SCHEDULES,
        "data_load_error": not BUS_SCHEDULES_TD and verify_gtfs_data_present(GTFS_DATA_DIR)

    }
    return render_template('index.html', **context)


@app.route('/calculate', methods=['POST'])
def calculate_api():
    data = request.get_json()
    app.logger.info(f"Received raw data in /calculate: {data}")

    if not data or 'leave_time' not in data:
        return jsonify({'error': 'Departure time (leave_time) not provided.'}), 400

    leave_time_str = data['leave_time']

    try:
        max_bus_delay_at_zoo_minutes = int(data.get('max_bus_delay_at_zoo_minutes', 0))
        max_bus_delay_at_zoo_minutes = max(0, min(max_bus_delay_at_zoo_minutes, MAX_INPUT_DELAY_MINUTES))
    except (ValueError, TypeError):
        max_bus_delay_at_zoo_minutes = 0
        app.logger.warning(
            f"Invalid value for max_bus_delay_at_zoo_minutes: {data.get('max_bus_delay_at_zoo_minutes')}. Using 0.")

    try:
        max_bus_delay_en_route_minutes = int(data.get('max_bus_delay_en_route_minutes', 0))
        max_bus_delay_en_route_minutes = max(0, min(max_bus_delay_en_route_minutes, MAX_INPUT_DELAY_MINUTES))
    except (ValueError, TypeError):
        max_bus_delay_en_route_minutes = 0
        app.logger.warning(
            f"Invalid value for max_bus_delay_en_route_minutes: {data.get('max_bus_delay_en_route_minutes')}. Using 0.")

    app.logger.info(
        f"API /calculate: leave_time='{leave_time_str}', "
        f"delay_at_zoo='{max_bus_delay_at_zoo_minutes}', delay_en_route='{max_bus_delay_en_route_minutes}'"
    )

    if MEETING_DATETIME_OBJ is None:
        app.logger.error("MEETING_DATETIME_OBJ is not set. Cannot calculate lateness.")
        return jsonify({'error': 'Server configuration error: Meeting time not set.'}), 503

    if not BUS_SCHEDULES_TD:
        app.logger.error("BUS_SCHEDULES_TD is empty. Cannot calculate lateness.")
        return jsonify({'error': 'Server error: Bus schedule data not available.'}), 503

    probability = calculate_single_lateness(
        leave_time_str, BUS_SCHEDULES_TD, MEETING_DATETIME_OBJ,
        WALK_HOME_TO_ZOO_SEC, WALK_TOOMPARK_TO_MEETING_SEC,
        max_bus_delay_at_zoo_minutes=max_bus_delay_at_zoo_minutes,
        max_bus_delay_en_route_minutes=max_bus_delay_en_route_minutes
    )

    if probability == -1.0:
        return jsonify({'error': f"Invalid time format for '{leave_time_str}'. Use HH:MM or HH:MM:SS."}), 400

    prob_percent = probability * 100
    if max_bus_delay_at_zoo_minutes == 0 and max_bus_delay_en_route_minutes == 0:
        status_message = "Rita will be LATE." if probability == 1.0 else "Rita will be ON TIME."
    else:
        if probability >= 0.99:
            status_message = f"Rita will almost certainly be LATE (P(Late) = {prob_percent:.0f}%)."
        elif probability >= 0.75:
            status_message = f"Rita is LIKELY to be LATE (P(Late) = {prob_percent:.0f}%)."
        elif probability >= 0.25:
            status_message = f"Rita MIGHT be late (P(Late) = {prob_percent:.0f}%)."
        elif probability > 0.001:
            status_message = f"Rita is very likely ON TIME, small chance of being late (P(Late) = {prob_percent:.1f}%)."
        else:
            status_message = f"Rita will almost certainly be ON TIME (P(Late) = {prob_percent:.0f}%)."

    is_late_flag_simple = (probability > 0.5)

    return jsonify({
        'leave_time_processed': leave_time_str,
        'probability': probability,
        'is_late': is_late_flag_simple,
        'status_message': status_message
    })


if __name__ == '__main__':
    initialize_app_data()
    app.logger.info("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=5001)
else:
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        if not os.path.exists(STATIC_FOLDER):
            os.makedirs(STATIC_FOLDER, exist_ok=True)
        initialize_app_data()