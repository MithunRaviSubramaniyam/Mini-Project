import cv2
import pickle

zoom_scale = 1.3
drawing = False
ix, iy = -1, -1
manual_mode = False
manual_first_point = None
current_image_index = 0

image_paths = ['carParkPortrait1.png', 'carParkPortrait2.png']

# Load Saved Positions
posLists = []
for i in range(2):
    try:
        with open(f'CarParkPos_{i}', 'rb') as f:
            posLists.append(pickle.load(f))
    except:
        posLists.append([])

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, manual_mode, manual_first_point, posLists, img

    # Convert zoomed mouse position back to original scale
    x = int(x / zoom_scale)
    y = int(y / zoom_scale)

    if manual_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            if manual_first_point is None:
                manual_first_point = (x, y)
                print("Manual: First point selected.")
            else:
                x1, y1 = manual_first_point
                x2, y2 = x, y
                top_left = (min(x1, x2), min(y1, y2))
                bottom_right = (max(x1, x2), max(y1, y2))
                posLists[current_image_index].append([top_left, bottom_right])
                with open(f'CarParkPos_{current_image_index}', 'wb') as f:
                    pickle.dump(posLists[current_image_index], f)
                print(f"Manual: Added box from {top_left} to {bottom_right}")
                manual_first_point = None
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_temp = img.copy()
            overlay = img_temp.copy()
            cv2.rectangle(overlay, (ix, iy), (x, y), (255, 200, 255), -1)
            cv2.addWeighted(overlay, 0.3, img_temp, 0.7, 0, img_temp)
            cv2.rectangle(img_temp, (ix, iy), (x, y), (255, 0, 255), 1)
            zoomed_temp = cv2.resize(img_temp, None, fx=zoom_scale, fy=zoom_scale)
            cv2.imshow("Image", zoomed_temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            posLists[current_image_index].append([(x1, y1), (x2, y2)])
            with open(f'CarParkPos_{current_image_index}', 'wb') as f:
                pickle.dump(posLists[current_image_index], f)
            print(f"Drag: Added box from ({x1}, {y1}) to ({x2}, {y2})")

while True:
    img = cv2.imread(image_paths[current_image_index])

    for i, rect in enumerate(posLists[current_image_index]):
        top_left, bottom_right = rect
        overlay = img.copy()
        cv2.rectangle(overlay, top_left, bottom_right, (200, 255, 200), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        # Show slot number
        cv2.putText(img, str(i + 1), (top_left[0] + 5, top_left[1] + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    if manual_mode and manual_first_point is not None:
        cv2.circle(img, manual_first_point, 4, (0, 0, 255), -1)
        cv2.putText(img, "Manual: Select 2nd point", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    zoomed_img = cv2.resize(img, None, fx=zoom_scale, fy=zoom_scale)
    cv2.imshow("Image", zoomed_img)
    cv2.setMouseCallback("Image", draw_rectangle)

    key = cv2.waitKey(1)

    if key == ord('b') and posLists[current_image_index]:
        removed = posLists[current_image_index].pop()
        with open(f'CarParkPos_{current_image_index}', 'wb') as f:
            pickle.dump(posLists[current_image_index], f)
        print(f"Deleted: {removed}")

    elif key == ord('c'):
        posLists[current_image_index] = []
        with open(f'CarParkPos_{current_image_index}', 'wb') as f:
            pickle.dump(posLists[current_image_index], f)
        print("Cleared all boxes.")

    elif key == ord('m'):
        manual_mode = not manual_mode
        manual_first_point = None
        print("Switched to Manual" if manual_mode else "Switched to Drag")

    elif key == ord('s'):
        current_image_index = 1 - current_image_index
        print(f"Switched to image: {image_paths[current_image_index]}")

    elif key == ord('p'):
        if current_image_index == 0:
            posLists[1] = posLists[0][:]
            with open('CarParkPos_1', 'wb') as f:
                pickle.dump(posLists[1], f)
            print("Copied boxes from image 0 to image 1.")
        else:
            posLists[0] = posLists[1][:]
            with open('CarParkPos_0', 'wb') as f:
                pickle.dump(posLists[0], f)
            print("Copied boxes from image 1 to image 0.")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()