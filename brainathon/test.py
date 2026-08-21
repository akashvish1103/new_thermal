import pandas as pd
import glob
import os


# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = r"C:\Users\Akash Vishwakarma\Desktop\new_thermal\brainathon\attendence_files"

OUTPUT_FILE = os.path.join(
    INPUT_FOLDER,
    "Master_Attendance.xlsx"
)


# ============================================================
# FIND ALL EXCEL FILES
# ============================================================

files = glob.glob(
    os.path.join(INPUT_FOLDER, "*.xlsx")
)

# Don't read the master file if it already exists
files = [
    f for f in files
    if os.path.basename(f) != "Master_Attendance.xlsx"
]

print("\nFiles found:")

for file in files:
    print("  ", os.path.basename(file))


# ============================================================
# POSSIBLE EMAIL COLUMN NAMES
# ============================================================

possible_email_columns = [
    "Registered E-Mail Address",
    "E-Mail",
    "Email Address",
    "Email",
    "E-mail",
    "E-mail Address"
]


# ============================================================
# READ ALL ATTENDANCE FILES
# ============================================================

all_attendance = []


for file in files:

    filename = os.path.basename(file)

    print("\nReading:", filename)

    df = pd.read_excel(file)

    # --------------------------------------------------------
    # CHECK NAME COLUMN
    # --------------------------------------------------------

    if "Name" not in df.columns:

        print(
            f"ERROR: 'Name' column not found in {filename}"
        )

        print("Available columns:")
        print(df.columns.tolist())

        continue

    # --------------------------------------------------------
    # FIND EMAIL COLUMN AUTOMATICALLY
    # --------------------------------------------------------

    email_column = None

    for column in possible_email_columns:

        if column in df.columns:
            email_column = column
            break

    if email_column is None:

        print(
            f"ERROR: No email column found in {filename}"
        )

        print("Available columns:")
        print(df.columns.tolist())

        continue

    print("Using email column:", email_column)

    # --------------------------------------------------------
    # SELECT NAME + EMAIL
    # --------------------------------------------------------

    temp = df[
        [
            "Name",
            email_column
        ]
    ].copy()

    # Rename columns
    temp.rename(
        columns={
            "Name": "Participant Name",
            email_column: "Email ID"
        },
        inplace=True
    )

    # --------------------------------------------------------
    # REMOVE EMPTY NAMES
    # --------------------------------------------------------

    temp.dropna(
        subset=["Participant Name"],
        inplace=True
    )

    # --------------------------------------------------------
    # CLEAN NAMES
    # --------------------------------------------------------

    temp["Participant Name"] = (
        temp["Participant Name"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # --------------------------------------------------------
    # CLEAN EMAILS
    # --------------------------------------------------------

    temp["Email ID"] = (
        temp["Email ID"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Remove empty / invalid-looking email values
    temp = temp[
        temp["Email ID"].notna()
    ]

    temp = temp[
        temp["Email ID"] != ""
    ]

    temp = temp[
        temp["Email ID"] != "nan"
    ]

    # --------------------------------------------------------
    # STORE DAY / FILE INFORMATION
    # --------------------------------------------------------

    temp["Attendance File"] = filename

    # Add records
    all_attendance.append(temp)


# ============================================================
# CHECK WHETHER ANY DATA WAS FOUND
# ============================================================

if not all_attendance:

    raise ValueError(
        "No valid attendance files were found."
    )


# ============================================================
# COMBINE ALL DAYS
# ============================================================

attendance = pd.concat(
    all_attendance,
    ignore_index=True
)


# ============================================================
# CREATE UNIQUE PARTICIPANT KEY
# ============================================================

# Email is used as the unique identifier.

attendance["_email_key"] = (
    attendance["Email ID"]
    .str.lower()
    .str.strip()
)


# ============================================================
# COUNT DAYS ATTENDED
# ============================================================

days_attended = (
    attendance
    .groupby("_email_key")
    .size()
    .reset_index(
        name="Days Attended"
    )
)


# ============================================================
# GET ONE RECORD PER PARTICIPANT
# ============================================================

master = (
    attendance
    .drop_duplicates(
        subset="_email_key",
        keep="first"
    )
)


# ============================================================
# ADD DAYS ATTENDED
# ============================================================

master = master.merge(
    days_attended,
    on="_email_key",
    how="left"
)


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

master = master[
    [
        "Participant Name",
        "Email ID",
        "Days Attended"
    ]
]


# ============================================================
# SORT ALPHABETICALLY
# ============================================================

master = master.sort_values(
    by="Participant Name",
    key=lambda x: x.str.lower()
).reset_index(drop=True)


# ============================================================
# ADD SERIAL NUMBER
# ============================================================

master.insert(
    0,
    "S.No.",
    range(
        1,
        len(master) + 1
    )
)


# ============================================================
# SAVE MASTER EXCEL
# ============================================================

master.to_excel(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("MASTER ATTENDANCE CREATED")
print("=" * 60)

print(
    f"Attendance files processed : {len(files)}"
)

print(
    f"Total attendance records   : {len(attendance)}"
)

print(
    f"Unique participants        : {len(master)}"
)

print(
    f"Output file                : {OUTPUT_FILE}"
)

print("\nMaster Attendance:")
print(master.to_string(index=False))