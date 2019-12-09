#pragma once
#include <random>

namespace tk { namespace gui {

    struct Color_t{
        uint8_t r,  g,  b,  a;
    };


    inline Color_t color4f(float r, float g, float b, float a) {
        return Color_t { uint8_t(r*255), uint8_t(g*255), uint8_t(b*255), uint8_t(a*255) };
    }


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


    inline Color_t randomColor(float alpha = 1.0) {
        Color_t cols[] = {
                color::RED,
                color::PINK,
                color::PURPLE,
                color::DEEP_PURPLE,
                color::INDIGO,
                color::BLUE,
                color::LIGHT_BLUE,
                color::CYAN,
                color::TEAL,
                color::GREEN,
                color::LIGHT_GREEN,
                color::LIME,
                color::YELLOW,
                color::AMBER,
                color::ORANGE,
                color::DEEP_ORANGE,
                color::WHITE,
        };
        int n_cols = sizeof(cols)/sizeof(*cols);

        static std::mt19937 rng(48);
        Color_t c = cols[rng() % n_cols];
        c.a = alpha*255;
        return c;
    }

    }} // namespace name
