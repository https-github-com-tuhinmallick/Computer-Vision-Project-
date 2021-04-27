import numpy as np
import cv2


class SeamCarver:
    def __init__(self, filename, out_height, out_width, protect_mask='', object_mask=''):
        # initialize parameter
        self.filename = filename
        self.out_height = out_height
        self.out_width = out_width

        # read in image and store as np.float64 format
        self.in_image = cv2.imread(filename).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)

        self.protect = (protect_mask != '')
        if self.protect:
          # if protect_mask filename is provided, read in protect mask image file as np.float64 format in gray scale
          self.mask = cv2.imread(protect_mask, 0).astype(np.float64)

        # kernel for forward energy map calculation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        # constant for covered area by protect mask 
        self.constant = 1000

        # starting seam carving function (image retargeting) will be process
        self.seams_carving()



    def seams_carving(self):
        """ 
        :return:
        We first process seam insertion or removal in vertical direction then followed by horizontal direction.
        If targeting height or width is greater than original ones --> seam insertion,
        else --> seam removal
        The algorithm is written for seam processing in vertical direction (column), so image is rotated 90 degree
        counter-clockwise for seam processing in horizontal direction (row)
        """
        #Exercise 4.1
        # calculate number of rows and columns needed to be inserted or removed
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)
        
        # remove column
        if delta_col < 0:
            self.seams_removal(delta_col * -1)
        # insert column
        elif delta_col > 0:
            self.seams_insertion(delta_col)

        # remove row
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, True)
            if self.protect:
                self.mask = self.rotate_mask(self.mask, 1)
            self.seams_removal(deltaWe_row * -1)
            self.out_image = self.rotate_image(self.out_image, False)
        # insert row
        elif delta_row > 0:
            self.out_image = self.rotate_image(self.out_image, True)
            if self.protect:
                self.mask = self.rotate_mask(self.mask, 1)
            self.seams_insertion(delta_row)
            self.out_image = self.rotate_image(self.out_image, False)



    def seams_removal(self, num_pixel):
        if self.protect:
            for dummy in range(num_pixel):
                energy_map = self.calc_energy_map()
                energy_map[np.where(self.mask > 0)] *= self.constant
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)
        else:
            for dummy in range(num_pixel):
                energy_map = self.calc_energy_map()
                cumulative_map = self.cumulative_map_forward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                self.delete_seam(seam_idx)


    def seams_insertion(self, num_pixel):
        if self.protect:
            temp_image = np.copy(self.out_image)
            temp_mask = np.copy(self.mask)
            seams_record = []

            for dummy in range(num_pixel):
                energy_map = self.calc_energy_map()
                energy_map[np.where(self.mask[:, :] > 0)] *= self.constant
                cumulative_map = self.cumulative_map_backward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                seams_record.append(seam_idx)
                self.delete_seam(seam_idx)
                self.delete_seam_on_mask(seam_idx)

            self.out_image = np.copy(temp_image)
            self.mask = np.copy(temp_mask)
            n = len(seams_record)
            for dummy in range(n):
                seam = seams_record.pop(0)
                self.add_seam(seam)
                self.add_seam_on_mask(seam)
                seams_record = self.update_seams(seams_record, seam)
        else:
            temp_image = np.copy(self.out_image)
            seams_record = []

            for dummy in range(num_pixel):
                energy_map = self.calc_energy_map()
                cumulative_map = self.cumulative_map_backward(energy_map)
                seam_idx = self.find_seam(cumulative_map)
                seams_record.append(seam_idx)
                self.delete_seam(seam_idx)

            self.out_image = np.copy(temp_image)
            n = len(seams_record)
            for dummy in range(n):
                seam = seams_record.pop(0)
                self.add_seam(seam)
                seams_record = self.update_seams(seams_record, seam)

    #TODO: change to laplacian
    def calc_energy_map(self):
        b, g, r = cv2.split(self.out_image)
        #https://www.tutorialspoint.com/opencv/opencv_scharr_operator.htm
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy

        """
        #https://www.tutorialspoint.com/opencv/opencv_laplacian_transformation.htm
        b_energy = np.absolute(cv2.Laplacian(b, 10)) + np.absolute(cv2.Laplacian(b, 10)) 
        g_energy = np.absolute(cv2.Laplacian(g, 10)) + np.absolute(cv2.Laplacian(g, 10))
        r_energy = np.absolute(cv2.Laplacian(r, 10)) + np.absolute(cv2.Laplacian(r, 10))
        return b_energy + g_energy + r_energy
        """


    def cumulative_map_backward(self, energy_map):
        m, n = energy_map.shape
        #Lets create a 2D array call output to store the minimum energy value seen upto that pixel.
        # output[i,j] will contain the smallest energy at that point in the image, considering all the possible seams upto that point from the top of the image.
        #So, the minimum energy required to traverse from the top of the image to bottom will be present in the last row of output. 
        #We need to backtrack from this to find the list of pixels present in this seam, so we’ll hold onto those values with a 2D array call backtrack.
        output = np.copy(energy_map)
        #Output[row,column]=energy_map[row, column]+ min(output(row-1, column-1),output(row-1, column),output(row-1, column+1))
        for row in range(1, m):
            for col in range(n):
                output[row, col] = \
                    energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
                    #https://numpy.org/doc/stable/reference/generated/numpy.amin.html
        return output

        """
        backtrack = np.zeros_like(output, dtype=np.int)

        for row in range(1, m):
          for col in range(0, n):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(output[row - 1, col:col + 2])
                backtrack[row, col] = idx + col
                min_energy = output[row - 1, idx + col]
            else:
                idx = np.argmin(output[row - 1, col - 1:col + 2])
                backtrack[row, col] = idx + col - 1
                min_energy = output[row - 1, idx + col - 1]

            output[row, col] += min_energy

        return output, backtrack
        """


    def cumulative_map_forward(self, energy_map):
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == n - 1:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return output


    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_image)
        #http://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output


    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        # Create a (r, c) matrix filled with the value False
        # We'll be removing all pixels from the image which
        # have True later
        output = np.zeros((m,), dtype=np.uint32)
        # Find the position of the smallest element in the
        # last row of output
        output[-1] = np.argmin(cumulative_map[-1])
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0:
                output[row] = np.argmin(cumulative_map[row, : 2])
            else:
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output


    def delete_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.out_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.out_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.out_image[row, :, 2], [col])
        self.out_image = np.copy(output)


    def add_seam(self, seam_idx):
        m, n = self.out_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.out_image[row, col: col + 2, ch])
                    output[row, col, ch] = self.out_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
                else:
                    p = np.average(self.out_image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = self.out_image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.out_image[row, col:, ch]
        self.out_image = np.copy(output)


    def update_seams(self, remaining_seams, current_seam):
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output

    def rotate_image(self, image, clockwise):
        k = 1 if clockwise else 3
        return np.rot90(image, k) 

    def rotate_mask(self, mask, clockwise):
        k = 1 if clockwise else 3
        return np.rot90(mask, k)


    def delete_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        output = np.zeros((m, n - 1))
        for row in range(m):
            col = seam_idx[row]
            output[row, : ] = np.delete(self.mask[row, : ], [col])
        self.mask = np.copy(output)


    def add_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        output = np.zeros((m, n + 1))
        for row in range(m):
            col = seam_idx[row]
            if col == 0:
                p = np.average(self.mask[row, col: col + 2])
                output[row, col] = self.mask[row, col]
                output[row, col + 1] = p
                output[row, col + 1: ] = self.mask[row, col: ]
            else:
                p = np.average(self.mask[row, col - 1: col + 1])
                output[row, : col] = self.mask[row, : col]
                output[row, col] = p
                output[row, col + 1: ] = self.mask[row, col: ]
        self.mask = np.copy(output)



    def save_result(self, filename):
        cv2.imwrite(filename, self.out_image.astype(np.uint8))
