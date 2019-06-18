import cv2
import time
import matplotlib.pyplot as plt

CLUSTER_THRESHOLD = 10
show = True

def detect_vertical_ife(image, rotation):
    
    print("Vertical detection")
    # En java se rotaba la imagen por un problema con pdfbox que metia rotaciones
    image_rotated = rotate_image(image,rotation)
    if show:
        plt.imshow(image_rotated, cmap='gray')
        plt.title("image_rotated")
        plt.show() 
    # Transform source image to gray if it is not
    image_gray = image_rotated
    if image.ndim == 3:
        image_gray = cv2.cvtColor(image_rotated,cv2.COLOR_RGB2GRAY) #COLOR_RGB2GRAY in java
    if show:
        plt.imshow(image_gray, cmap='gray')
        plt.title("image_gray")
        plt.show() 
    # Apply bilateral filter to reduce noise and preserve edges
    image_bilateral = cv2.bilateralFilter(image_gray,11,17,17) #FIX
    if show:
        plt.imshow(image_bilateral, cmap='gray')
        plt.title("image_bilateral")
        plt.show()
    # Sobel edge detection
    image_sobeled = sobel_detection(image_bilateral,5)
    if show:
        plt.imshow(image_sobeled, cmap='gray')
        plt.title("image_sobeled")
        plt.show()
    # Thresholding image
    _, image_thresholded = cv2.threshold(image_sobeled,50,255,cv2.THRESH_BINARY)
    if show:
        plt.imshow(image_thresholded, cmap='gray')
        plt.title("image_thresholded")
        plt.show()
    # Morphology operations
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1)) #ASK Cross with size 1x1 pixel ??? 1 insted of SHAPE_CROSS no shape cross on python?
    #TODO manejar si solo hay un contorno
    image_first_morph = cv2.morphologyEx(image_thresholded,cv2.MORPH_CLOSE,cross) #Closing is the same as making a dilation followed by a erosion
    if show:
        plt.imshow(image_first_morph, cmap='gray')
        plt.title("image_first_morph")
        plt.show()
    # Erode and then dilate is the same as:
    #cv2.morphologyEx(image_first_morph,cv2.MORPH_OPEN,cross)
    image_second_morph = cv2.erode(image_first_morph,cross,3) # what means cv2.erode(image_first_morph,cross,(-1,-1),3) (-1,-1)
    if show:
        plt.imshow(image_second_morph, cmap='gray')
        plt.title("image_second_morph")
        plt.show()
    image_third_morph = cv2.dilate(image_second_morph,cross,3)
    if show:
        plt.imshow(image_third_morph, cmap='gray')
        plt.title("image_third_morph")
        plt.show()
    #Find contours
    _, contours, hierarchy = cv2.findContours(image_third_morph,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    #Filter the small areas with the incorrect width/height for main contour and sort by descending area
    size_between = lambda contour, small, big : small > cv2.boundingRect(contour)[2]/cv2.boundingRect(contour)[3] < big # Nothing built-in python?
    size_compared_to_image = lambda contour, image : cv2.boundingRect(contour)[2] > 0.2 * image.shape[1] #ASK The detection of the document is based on a proportion with the main image
    print(f"number of contours {len(contours)}")
    #Gets the biggest contour
    #FIX not working
    contours_filtered_and_sorted = sorted([contour for contour in contours if size_between(contour, 1.5, 1.8) and size_compared_to_image(contour, image_third_morph)],reverse=True,key=contour_area)
    print(f"numero de contornos validos {len(contours_filtered_and_sorted)}")
    if contours_filtered_and_sorted:
        main_contour = contours_filtered_and_sorted[0]
        #Gets his location
        main_contour_x,main_contour_y,main_contour_w,main_contour_h = cv2.boundingRect(main_contour)
        #Crops the image_bilateral to get an image with just the document
        #ASK this is similar to somethig that sergio did
        document_image = image_bilateral[main_contour_y : main_contour_y + main_contour_h][main_contour_x : main_contour_x + main_contour_w]
        if show:
            plt.imshow(document_image, cmap='gray')
            plt.title("document_image")
            plt.show()
        #ASK FIXED_CONTOUR_SIZE

    # Detect the numbers inside the document image
    numbers_image_sobeled =sobel_detection(document_image,-1)
    if show:
            plt.imshow(numbers_image_sobeled, cmap='gray')
            plt.title("numbers_image_sobeled")
            plt.show()
    _, numbers_image_thresholded = cv2.threshold(numbers_image_sobeled,30,255,cv2.THRESH_BINARY)

    #main_contour_area = main_contour_h * main_contour_w
    #RETR_EXTERNAL
    _, contours, _ = cv2.findContours(numbers_image_thresholded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print(f"number of contours fuck {len(contours)}")
    cv2.drawContours(document_image, contours, -1, (255, 255, 255), 3)
    cv2.circle(document_image,(50,50), radius=20,color=(255,255,255),thickness=3)

    if show:
        #RGB_img = cv2.cvtColor(document_image, cv2.COLOR_BGR2RGB)
        plt.imshow(document_image)
        plt.title("document_image")
        plt.show()



    #!!contour_compare_to_main = 0.00005 > contour/main_contour_area < 0.3
    #!!thrid_condition = contour.x < main_contour_w / 4 #ASK que es contour x
    contour_compare_to_main = True
    thrid_condition = True

    contours_filtered_and_sorted = sorted([contour for contour in contours if size_between(contour,1.1,1.7) and contour_compare_to_main and thrid_condition],reverse=True, key=contour_area)
    print(f"number of contours_filtered_and_sorted {len(contours_filtered_and_sorted)}")
    
    contours_without_inside_boxes = remove_inside_boxes(contours_filtered_and_sorted)
    print(f"number of contours_without_inside_boxes {len(contours_without_inside_boxes)}")
    
    clustered_contours = cluster_y(contours_without_inside_boxes)
    print(f"number of clustered_contours {len(clustered_contours)}")

    final_IFE = [contour for contour in clustered_contours if size_between(contour,0.09,0.17) and cv2.boundingRect(contour)[3] > main_contour_h * 0.3]
    print(f"number of final_IFE {len(final_IFE)}")
    if final_IFE:
        print("Estoy aqui")
        #Solo hay que hacer parte ya que la mayoria es para integrar este texto en la imagen
        pass




def rotate_image(image,rotation):
    flip_code = lambda rotation : 0 if rotation == 270 else (1 if  rotation == 180  else (-1 if rotation == 90 else None))
    print(flip_code(rotation))
    image_transpose = cv2.transpose(image)
    flip_image = cv2.flip(image_transpose, flip_code(rotation))

    return flip_image

# Doesn't hierarchy do this?
def remove_inside_boxes(contours):
    '''
     from a list of contours returns a list with only the external contourns
    '''
    #MAYBE cv2.matchShapes()
    #ASK para que es el sorted con comparingInt
    return [conotur for conotur in contours if not is_inside(conotur,contours)]

#Shouldn't be called is_inside?
def is_inside(contour_to_compare,contours):
    '''
    Check if a contour is inside other or if it's equal over a list of contours
    '''
    #CLEAN
    contour_to_compare_rect = cv2.boundingRect(contour_to_compare)
    #contour_to_compare_x, contour_to_compare_y, contour_to_compare_w, contour_to_compare_h = contour_to_compare.boundingRect()
    rect_is_equal = lambda contour_to_compare_rect, contour_rect : contour_to_compare_rect == contour_rect
    #CLEAN maybe simplify down lambda
    rect_is_inside = lambda contour_to_compare_rect, contour : contour_to_compare_rect[0] > contour[0] and contour_to_compare_rect[1] > contour[1] and contour_to_compare_rect[0] + contour_to_compare_rect[2] < contour[0] + contour[2] and contour_to_compare_rect[1] + contour_to_compare_rect[3] < contour[1] + contour[3]
    for contour in contours:
        if rect_is_equal(contour_to_compare_rect,cv2.boundingRect(contour)) or rect_is_inside(contour_to_compare_rect,cv2.boundingRect(contour)):
            return True
    

def cluster_y(contours):
    '''
    Cluster boxes with similar X values and low Y difference
    '''
    #Loops through all the contours and call merge_y for each
    #for contour in contours:
    #    result.append(merge_y(contour, contours))

    return [merge_y(contour, contours)for contour in contours]


def merge_y(contour_to_compare,contours,result):
    '''
    Nothing built-in python or opencv?
    '''
    #Duplicated
    contour_to_compare_rect = contour_to_compare.boundingRect()

    rect_is_equal = lambda contour_to_compare_rect,contour : contour_to_compare_rect == contour.boundingRect()
    abs_less_than_threshold = lambda value : abs(value) < CLUSTER_THRESHOLD
    close_y = lambda contour_to_compare_rect, contour_y : contour_to_compare_rect[1] + contour_to_compare_rect[3] -  contour_y
    close_x = lambda contour_to_compare_rect,contour_x :  contour_to_compare_rect - contour_x

    close_contours = [contour for contour in contours if rect_is_equal(contour_to_compare_rect,contour) and close_x and close_y]#.!!map(merge_boxes())

    rect_list = [merge_boxes(contour, contour_to_compare) for contour in close_contours]
    if rect_list:
        merged = rect_list[0]
        if not result:
            result.append.get(0)
        else:
            close_result = [contour for contour in result if contour.boundingRect() == merged.boundingRect() and abs(merged.boundingRect[1] + merged.boundingRect[3] - contour.boundingRect[1]) > CLUSTER_THRESHOLD and abs(contour_to_compare_rect[0] - contour.boundingRect[0]) < CLUSTER_THRESHOLD ]
            if not close_result:
                result.append(merged)
            else:
                merged_rect = merge_boxes(merged,close_result[0])
                result.remove(close_result[0])
                result.append(merged_rect)

    return result  #???
    
def merge_boxes(contour_one, contour_two):
    contour_one_x, contour_one_y, contour_one_w, contour_one_h = contour_one.boundingRect()
    contour_two_x, contour_two_y, contour_two_w, contour_two_h = contour_two.boundingRect()
    start = max(contour_one_x, contour_two_x)
    end = max(contour_one_y + contour_one_h, contour_two_y + contour_two_h)

    return (start, end) #???

def sobel_detection(imagen,ksize):
    
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize)
    if show:
        plt.imshow(sobel_x, cmap='gray')
        plt.title("sobel_x")
        plt.show()
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize)
    if show:
        plt.imshow(sobel_y, cmap='gray')
        plt.title("sobel_y")
        plt.show()
    subtract = cv2.subtract(sobel_x, sobel_y)
    if show:
        plt.imshow(subtract, cmap='gray')
        plt.title("subtract")
        plt.show()
    result = cv2.convertScaleAbs(subtract)
    if show:
        plt.imshow(result, cmap='gray')
        plt.title("result")
        plt.show()
    
    return result 

def contour_area(contour):
    
    _, _, w, h = cv2.boundingRect(contour)
    return w * h


    


    
        



def main():
    
    fake_image = cv2.imread('Ejemplo2.png')
    detect_vertical_ife(fake_image,180)

if __name__ == '__main__':
    main()