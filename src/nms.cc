#include "../include/nms.h"

NMS::NMS(size_t window_size) : window_size_(window_size), n_((window_size - 1) / 2) {}

// for (const auto& point : points) {
//     if (isInBoarder(point, i, j, i + n, j + n)) {
//         if (image.at<u_char>(static_cast<int>(point.x), static_cast<int>(point.y)) >
//             image.at<u_char>(begin_index_x, begin_index_y)) {
//             begin_index_x = static_cast<int>(point.x);
//             begin_index_y = static_cast<int>(point.y);
//         }
//     }
// }