#pragma once

namespace tk { namespace gui {
    struct Color_t{
        uint8_t r,  g,  b,  a;
    };

    namespace color {
        static Color_t RED          = {244, 67, 54, 255};
        static Color_t PINK         = {233, 30, 99, 255};
        static Color_t PURPLE       = {156, 39, 176, 255};
        static Color_t DEEP_PURPLE  = {103, 58, 183, 255};
        static Color_t INDIGO       = {63, 81, 181, 255};
        static Color_t BLUE         = {33, 150, 243, 255};
        static Color_t LIGHT_BLUE   = {3, 169, 244, 255};
        static Color_t CYAN         = {0, 188, 212, 255};
        static Color_t TEAL         = {0, 150, 136, 255};
        static Color_t GREEN        = {76, 175, 80, 255};
        static Color_t LIGHT_GREEN  = {139, 195, 74, 255};
        static Color_t LIME         = {205, 220, 57, 255};
        static Color_t YELLOW       = {255, 235, 59, 255};
        static Color_t AMBER        = {255, 193, 7, 255};
        static Color_t ORANGE       = {255, 152, 0, 255};
        static Color_t DEEP_ORANGE  = {255, 87, 34, 255};
        static Color_t BROWN        = {121, 85, 72, 255};
        static Color_t GREY         = {158, 158, 158, 255};
        static Color_t BLUE_GRAY    = {96, 125, 139, 255};
        static Color_t DARK_GRAY    = {33, 33, 33, 255};
        static Color_t BLACK        = {0, 0, 0, 255};
        static Color_t WHITE        = {250, 250, 250, 255};
    }    
}} // namespace name