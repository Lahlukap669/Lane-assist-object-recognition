def HughToLineMask(points, hughLinesPResult, widthOfPhoto):
    xyHough = []
    for point in points:
        sdist = 10000
        cur_xy = []
        ##LEVA STRAN
        for line in hughLinesPResult[0]:
            dist1 = abs(point[0] - line[0]) + abs(point[1] - line[1])
            dist2 = abs(point[0] - line[2]) + abs(point[1] - line[3])
            if dist1 < sdist:
                sdist=dist1
                cur_xy.clear()
                cur_xy.append(line[0])
                cur_xy.append(line[1])
            if dist2 < sdist:
                sdist=dist2
                cur_xy.clear()
                cur_xy.append(line[2])
                cur_xy.append(line[3])
        xyHough.append(cur_xy)

    ##Maska
    slika1 = np.copy(slika)
                #3 channels
    prazna_slika = np.zeros((slika1.shape[0], slika1.shape[1], 3), dtype=np.uint8)
    cv2.line(prazna_slika, (xyHough[0][0], xyHough[0][1]), (xyHough[1][0], xyHough[1][1]), (255, 100, 50), thickness=2)
    cv2.line(prazna_slika, (xyHough[2][0], xyHough[2][1]), (xyHough[3][0], xyHough[3][1]), (255, 100, 50), thickness=2)
    slika1 = cv2.addWeighted(slika1, 0.6, prazna_slika, 1, 0.0)
    return slika1    
    
    
