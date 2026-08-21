# import os
# import re
# import shutil


# # ============================================================
# # 1. SOURCE FOLDER
# # ============================================================
# # CHANGE THIS PATH
# # This is the main folder containing all subject folders:
# #
# # 44_2026-07-07
# # 48_2026-07-08
# # 53_2026-07-10
# # etc.
# #
# SOURCE_ROOT = r"D:\Tihar_thermal_data_Input\Sabarmati_sample_data"


# # ============================================================
# # 2. DESTINATION FOLDER
# # ============================================================
# # CHANGE THIS PATH
# #
# # The script will automatically create:
# #
# # DESTINATION_ROOT
# #     ├── 44
# #     ├── 48
# #     ├── 53
# #     ├── 58
# #     └── ...
# #
# DESTINATION_ROOT = r"E:\sorted_data"


# # ============================================================
# # 3. VIDEO EXTENSIONS
# # ============================================================
# # Add/remove extensions here if required.

# VIDEO_EXTENSIONS = {
#     ".mpg",
#     ".mpeg",
#     ".mp4",
#     ".wmv",
#     ".avi",
#     ".mov"
# }


# # ============================================================
# # 4. GET SUBJECT ID
# # ============================================================

# def get_subject_id(folder_name):

#     """
#     Example:

#         44_2026-07-07  ->  44
#         48_2026-07-08  ->  48
#         61_2026-07-13  ->  61

#     """

#     match = re.match(r"^(\d+)_", folder_name)

#     if match:
#         return match.group(1)

#     return None


# # ============================================================
# # 5. FIND REQUIRED FILES
# # ============================================================

# def find_required_files(subject_folder, subject_id):

#     passive_video = None
#     passive_marker = None

#     hdrs_video = None
#     hdrs_marker = None


#     # --------------------------------------------------------
#     # Search recursively inside the subject folder
#     #
#     # This means it will look inside:
#     #
#     # 44_2026-07-07
#     #     ├── Passive
#     #     ├── HCDCI
#     #     └── HDRS
#     #
#     # We only select PASSIVE and HDRS files.
#     # --------------------------------------------------------

#     for root, dirs, files in os.walk(subject_folder):

#         for filename in files:

#             lower_name = filename.lower()
#             full_path = os.path.join(root, filename)


#             # =================================================
#             # PASSIVE THERMAL VIDEO
#             # =================================================

#             if (
#                 lower_name.startswith(
#                     f"{subject_id}_passive_thermal_"
#                 )
#                 and os.path.splitext(filename)[1].lower()
#                 in VIDEO_EXTENSIONS
#             ):

#                 passive_video = full_path


#             # =================================================
#             # PASSIVE MARKERS
#             # =================================================

#             elif lower_name in {
#                 f"{subject_id}_passive_markers.xlsx",
#                 f"{subject_id}_passive_markers.xls",
#                 f"{subject_id}_passive_markers.csv"
#             }:

#                 passive_marker = full_path


#             # =================================================
#             # HDRS THERMAL VIDEO
#             # =================================================

#             elif (
#                 lower_name.startswith(
#                     f"{subject_id}_hdrs_thermal_"
#                 )
#                 and os.path.splitext(filename)[1].lower()
#                 in VIDEO_EXTENSIONS
#             ):

#                 hdrs_video = full_path


#             # =================================================
#             # HDRS MARKERS
#             # =================================================

#             elif lower_name in {
#                 f"{subject_id}_hdrs_markers.xlsx",
#                 f"{subject_id}_hdrs_markers.xls",
#                 f"{subject_id}_hdrs_markers.csv"
#             }:

#                 hdrs_marker = full_path


#     return (
#         passive_video,
#         passive_marker,
#         hdrs_video,
#         hdrs_marker
#     )


# # ============================================================
# # 6. COPY FILE
# # ============================================================

# def copy_file(source_file, destination_folder):

#     if source_file is None:
#         return False

#     os.makedirs(destination_folder, exist_ok=True)

#     destination_file = os.path.join(
#         destination_folder,
#         os.path.basename(source_file)
#     )

#     shutil.copy2(
#         source_file,
#         destination_file
#     )

#     print(
#         f"Copied: {os.path.basename(source_file)}"
#     )

#     return True


# # ============================================================
# # 7. MAIN
# # ============================================================

# print("\n==============================================")
# print("STARTING FILE COLLECTION")
# print("==============================================\n")


# # Check source
# if not os.path.isdir(SOURCE_ROOT):

#     raise FileNotFoundError(
#         f"Source folder does not exist:\n{SOURCE_ROOT}"
#     )


# # Create destination
# os.makedirs(
#     DESTINATION_ROOT,
#     exist_ok=True
# )


# # Counters
# total_subjects = 0
# complete_subjects = 0
# incomplete_subjects = 0


# # ============================================================
# # LOOP THROUGH ALL SUBJECT FOLDERS
# # ============================================================

# for folder_name in os.listdir(SOURCE_ROOT):

#     subject_folder = os.path.join(
#         SOURCE_ROOT,
#         folder_name
#     )


#     # Only process folders
#     if not os.path.isdir(subject_folder):
#         continue


#     # Get subject ID
#     subject_id = get_subject_id(folder_name)


#     if subject_id is None:

#         print(
#             f"Skipping folder (subject ID not found): "
#             f"{folder_name}"
#         )

#         continue


#     total_subjects += 1


#     print("\n----------------------------------------------")
#     print(f"Subject: {subject_id}")
#     print(f"Source:  {folder_name}")
#     print("----------------------------------------------")


#     # ========================================================
#     # FIND FILES
#     # ========================================================

#     (
#         passive_video,
#         passive_marker,
#         hdrs_video,
#         hdrs_marker
#     ) = find_required_files(
#         subject_folder,
#         subject_id
#     )


#     # ========================================================
#     # DESTINATION SUBJECT FOLDER
#     # ========================================================

#     subject_destination = os.path.join(
#         DESTINATION_ROOT,
#         subject_id
#     )


#     # ========================================================
#     # COPY PASSIVE VIDEO
#     # ========================================================

#     if passive_video:

#         copy_file(
#             passive_video,
#             subject_destination
#         )

#     else:

#         print(
#             "MISSING: Passive Thermal Video"
#         )


#     # ========================================================
#     # COPY PASSIVE MARKERS
#     # ========================================================

#     if passive_marker:

#         copy_file(
#             passive_marker,
#             subject_destination
#         )

#     else:

#         print(
#             "MISSING: Passive Markers"
#         )


#     # ========================================================
#     # COPY HDRS VIDEO
#     # ========================================================

#     if hdrs_video:

#         copy_file(
#             hdrs_video,
#             subject_destination
#         )

#     else:

#         print(
#             "MISSING: HDRS Thermal Video"
#         )


#     # ========================================================
#     # COPY HDRS MARKERS
#     # ========================================================

#     if hdrs_marker:

#         copy_file(
#             hdrs_marker,
#             subject_destination
#         )

#     else:

#         print(
#             "MISSING: HDRS Markers"
#         )


#     # ========================================================
#     # CHECK WHETHER ALL 4 FILES EXISTED
#     # ========================================================

#     files_found = [
#         passive_video,
#         passive_marker,
#         hdrs_video,
#         hdrs_marker
#     ]

#     if all(file is not None for file in files_found):

#         complete_subjects += 1

#         print(
#             f"\n✓ Subject {subject_id}: "
#             f"ALL 4 FILES COPIED"
#         )

#     else:

#         incomplete_subjects += 1

#         print(
#             f"\n⚠ Subject {subject_id}: "
#             f"SOME FILES ARE MISSING"
#         )


# # ============================================================
# # FINAL SUMMARY
# # ============================================================

# print("\n\n==============================================")
# print("FILE COLLECTION COMPLETE")
# print("==============================================")

# print(
#     f"Total subjects found:       {total_subjects}"
# )

# print(
#     f"Subjects with all 4 files:  {complete_subjects}"
# )

# print(
#     f"Subjects with missing data: {incomplete_subjects}"
# )

# print(
#     f"\nDestination:\n{DESTINATION_ROOT}"
# )

# print("==============================================")

###################################################################################################################


import os
import re
import shutil


# ============================================================
# 1. VIDEO SOURCE FOLDER
# ============================================================
# CHANGE THIS PATH
#
# This is the folder containing ALL the thermal videos:
#
# 49_Passive_Thermal_30_40.mpg
# 50_HDRS_Thermal_30_40.mpg
# 50_Passive_Thermal_30_40.mpg
# ...
#
VIDEO_SOURCE_ROOT = r"D:\Tihar_thermal_data_Input\Exported_Thermal_IRSoft_Sabarmati"


# ============================================================
# 2. EXISTING SUBJECT-WISE DESTINATION FOLDER
# ============================================================
# CHANGE THIS PATH
#
# This is the folder you created previously:
#
# Destination
# ├── 44
# ├── 48
# ├── 53
# ├── ...
#
# The marker files are already inside these folders.
#
DESTINATION_ROOT = r"E:\sorted_data"


# ============================================================
# 3. VIDEO EXTENSIONS
# ============================================================

VIDEO_EXTENSIONS = {
    ".mpg",
    ".mpeg",
    ".mp4",
    ".wmv",
    ".avi",
    ".mov"
}


# ============================================================
# 4. EXTRACT SUBJECT ID FROM VIDEO NAME
# ============================================================

def get_subject_id(filename):

    """
    Examples:

        49_Passive_Thermal_30_40.mpg
            -> 49

        50_HDRS_Thermal_30_40.mpg
            -> 50

        61_Passive_Thermal_25_40.mpg
            -> 61
    """

    match = re.match(
        r"^(\d+)_(Passive|HDRS)_Thermal_",
        filename,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# 5. CHECK SOURCE FOLDER
# ============================================================

if not os.path.isdir(VIDEO_SOURCE_ROOT):

    raise FileNotFoundError(
        f"Video source folder does not exist:\n"
        f"{VIDEO_SOURCE_ROOT}"
    )


# ============================================================
# 6. CHECK DESTINATION FOLDER
# ============================================================

if not os.path.isdir(DESTINATION_ROOT):

    raise FileNotFoundError(
        f"Destination folder does not exist:\n"
        f"{DESTINATION_ROOT}"
    )


# ============================================================
# 7. FIND ALL VIDEOS
# ============================================================

video_files = []

for filename in os.listdir(VIDEO_SOURCE_ROOT):

    full_path = os.path.join(
        VIDEO_SOURCE_ROOT,
        filename
    )

    # Must be a file
    if not os.path.isfile(full_path):
        continue

    # Check extension
    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in VIDEO_EXTENSIONS:
        continue

    # Check whether it is one of our thermal videos
    subject_id = get_subject_id(filename)

    if subject_id is None:
        continue

    video_files.append(
        (filename, full_path, subject_id)
    )


# ============================================================
# 8. SORT VIDEOS
# ============================================================

video_files.sort(
    key=lambda x: (
        int(x[2]),
        x[0].lower()
    )
)


# ============================================================
# 9. COPY VIDEOS INTO SUBJECT FOLDERS
# ============================================================

copied = 0
missing_subject_folder = 0
skipped_existing = 0


print("\n")
print("=" * 70)
print("COPYING THERMAL VIDEOS INTO SUBJECT FOLDERS")
print("=" * 70)


for filename, source_path, subject_id in video_files:

    print("\n----------------------------------------------")

    print(
        f"Video:   {filename}"
    )

    print(
        f"Subject: {subject_id}"
    )


    # ========================================================
    # SUBJECT DESTINATION
    # ========================================================

    subject_folder = os.path.join(
        DESTINATION_ROOT,
        subject_id
    )


    # --------------------------------------------------------
    # If subject folder doesn't exist
    # --------------------------------------------------------

    if not os.path.isdir(subject_folder):

        print(
            f"WARNING: Subject folder does not exist:"
        )

        print(
            f"         {subject_folder}"
        )

        missing_subject_folder += 1

        continue


    # ========================================================
    # DESTINATION FILE
    # ========================================================

    destination_path = os.path.join(
        subject_folder,
        filename
    )


    # ========================================================
    # IF FILE ALREADY EXISTS
    # ========================================================

    if os.path.exists(destination_path):

        print(
            "SKIPPED: Video already exists in destination."
        )

        skipped_existing += 1

        continue


    # ========================================================
    # COPY
    # ========================================================

    print(
        "Copying..."
    )

    shutil.copy2(
        source_path,
        destination_path
    )

    print(
        "COPIED successfully."
    )

    copied += 1


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("COPY COMPLETE")
print("=" * 70)

print(
    f"Videos found:                 {len(video_files)}"
)

print(
    f"Videos copied:                {copied}"
)

print(
    f"Already existed / skipped:    {skipped_existing}"
)

print(
    f"Missing subject folders:      {missing_subject_folder}"
)

print("\nDestination:")
print(DESTINATION_ROOT)

print("=" * 70)