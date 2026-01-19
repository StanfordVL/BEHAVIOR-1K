from PIL import Image
import pathlib
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def distance_from_grayscale_stats(filename):
    img = Image.open(filename)
    arr = np.array(img)

    # Get the distance from grayscale for each pixel
    distances = np.sqrt(
        (
            (arr[:, :, 0] - arr[:, :, 1]) ** 2
            + (arr[:, :, 0] - arr[:, :, 2]) ** 2
            + (arr[:, :, 1] - arr[:, :, 2]) ** 2
        )
        / 3
    )

    # Get the min, max, mean, median, and 25th / 75th percentiles
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    mean_dist = np.mean(distances)
    dist_std = np.std(distances)
    median_dist = np.median(distances)
    q25_dist = np.percentile(distances, 25)
    q75_dist = np.percentile(distances, 75)

    return {
        "min": min_dist,
        "max": max_dist,
        "mean": mean_dist,
        "std": dist_std,
        "median": median_dist,
        "q25": q25_dist,
        "q75": q75_dist,
    }


def main():
    files = list(
        pathlib.Path(r"/scr/BEHAVIOR-1K/asset_pipeline").glob(
            "cad/*/*/bakery/*Reflection*.png"
        )
    )
    print("Total files:", len(files))

    # Go through all the files and get all the unique colors in each of them
    stats_by_file = {}
    with ProcessPoolExecutor() as executor:
        futures = {}

        for fn in tqdm(files, desc="Queueing files"):
            futures[executor.submit(distance_from_grayscale_stats, fn)] = fn

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            fn = futures[future]
            stats = future.result()
            stats_by_file[fn] = stats

    # Save the stats to a CSV file
    with open("reflection_analysis.csv", "w") as f:
        f.write("filename,min,max,mean,std,median,q25,q75\n")
        for fn, stats in stats_by_file.items():
            f.write(
                f"{fn},{stats['min']},{stats['max']},{stats['mean']},{stats['std']},{stats['median']},{stats['q25']},{stats['q75']}\n"
            )


if __name__ == "__main__":
    main()
