import numpy as np
import cv2


def classify_circles(rgb_image, circles):
    hsv_image = cv2.cvtColor(
        cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2HSV
    )
    classifications = []
    H, W = rgb_image.shape[:2]

    for (center_x, center_y, radius) in circles:
        # Build a circular mask
        row_idx, col_idx = np.ogrid[:H, :W]
        squared_distance = (col_idx - center_x) ** 2 + \
            (row_idx - center_y) ** 2
        inner_circle_mask = squared_distance <= (
            radius * 0.8) ** 2   # slightly smaller to avoid border

        masked_hsv_pixels = hsv_image[inner_circle_mask]
        if masked_hsv_pixels.size == 0:
            classifications.append("UNLIT")
            continue

        avg_hue = masked_hsv_pixels[:, 0].mean()   # 0–179 HSV
        avg_saturation = masked_hsv_pixels[:, 1].mean()   # 0–255
        avg_brightness = masked_hsv_pixels[:, 2].mean()   # 0–255

        masked_rgb_pixels = rgb_image[inner_circle_mask]
        avg_red = masked_rgb_pixels[:, 0].mean()
        avg_green = masked_rgb_pixels[:, 1].mean()
        avg_blue = masked_rgb_pixels[:, 2].mean()

        print(f"      Circle ({center_x},{center_y},r={radius}): "
              f"H={avg_hue:.1f} S={avg_saturation:.1f} V={avg_brightness:.1f} | "
              f"R={avg_red:.1f} G={avg_green:.1f} B={avg_blue:.1f}")

        # Classification logic
        # UNLIT: dark, low-saturation pixels
        if avg_brightness < 80 and avg_saturation < 60:
            classification = "UNLIT"
        elif avg_saturation < 40 and avg_brightness < 100:
            classification = "UNLIT"

        # RED: hue 0-20 or 160-179 in HSV, OR dominant R in RGB
        # Saturation > 50 and brightness > 80 -> ensure vivid color
        elif (avg_hue < 20 or avg_hue > 160) and avg_saturation > 50 and avg_brightness > 80:
            # Distinguish red from yellow using RGB: red has much more R than G
            if avg_red > avg_green * 1.3:
                classification = "RED"
            # Hue 0–12 or 165–179 -> typically red
            elif avg_hue < 12 or avg_hue > 165:
                classification = "RED"
            else:
                classification = "YELLOW"

        elif 20 <= avg_hue <= 35 and avg_saturation > 50 and avg_brightness > 80:
            # Hue 20-35 can be either yellow or a glowing red.
            # Red lights often shift hue toward orange/yellow when bright.
            # Distinguish via RGB: if R >> G, it's red not yellow.
            if avg_red > avg_green * 2.5 and avg_red > 150:
                classification = "RED"
            else:
                classification = "YELLOW"

        elif 35 < avg_hue < 95 and avg_saturation > 30 and avg_brightness > 80:
            # Bright red lights can shift hue into this range (H≈35-50).
            # If R is overwhelmingly dominant, it's red not green.
            if avg_red > avg_green * 2.5 and avg_red > avg_blue * 2.5 and avg_red > 150:
                classification = "RED"
            else:
                classification = "GREEN"

        # Fallback rules based on RGB dominance:
            # Red is strong -> "RED"
            # Green is strong -> "GREEN"
            # Otherwise -> "UNLIT"
        elif avg_red > avg_green * 1.5 and avg_red > avg_blue and avg_red > 120:
            classification = "RED"
        elif avg_green > avg_red and avg_green > avg_blue and avg_green > 100:
            classification = "GREEN"
        else:
            classification = "UNLIT"

        classifications.append(classification)
    return classifications
