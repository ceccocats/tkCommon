#pragma once
#include <random>

namespace tk { namespace gui {

    class Color_t{
        public:
            float color[4];
            void set(float r, float g, float b, float a){
                color[0] = r;
                color[1] = g;
                color[2] = b;
                color[3] = a;
            }
            void set(uint8_t r, uint8_t g, uint8_t b, uint8_t a){
                color[0] = r/255.0f;
                color[1] = g/255.0f;
                color[2] = b/255.0f;
                color[3] = a/255.0f;
            }
            float& r() {return color[0];}
            float& g() {return color[1];}
            float& b() {return color[2];}
            float& a() {return color[3];}
    };


    inline Color_t color4f(float r, float g, float b, float a) {
        return Color_t { r , g , b , a };
    }

    inline Color_t color4u(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        return Color_t { float(r/255.0), float(g/255.0), float(b/255.0), float(a/255.0) };
    }


    namespace color {
        static Color_t RED          = {244.0/255.0, 67.0/255.0,     54.0/255.0,     1.0};
        static Color_t PINK         = {233.0/255.0, 30.0/255.0,     99.0/255.0,     1.0};
        static Color_t PURPLE       = {156.0/255.0, 39.0/255.0,     176.0/255.0,    1.0};
        static Color_t DEEP_PURPLE  = {103.0/255.0, 58.0/255.0,     183.0/255.0,    1.0};
        static Color_t INDIGO       = {63.0/255.0,  81.0/255.0,     181.0/255.0,    1.0};
        static Color_t BLUE         = {33.0/255.0,  150.0/255.0,    243.0/255.0,    1.0};
        static Color_t LIGHT_BLUE   = {3.0/255.0,   169.0/255.0,    244.0/255.0,    1.0};
        static Color_t CYAN         = {0.0,         188.0/255.0,    212.0/255.0,    1.0};
        static Color_t TEAL         = {0.0,         150.0/255.0,    136.0/255.0,    1.0};
        static Color_t GREEN        = {76.0/255.0,  175.0/255.0,    80.0/255.0,     1.0};
        static Color_t LIGHT_GREEN  = {139.0/255.0, 195.0/255.0,    74.0/255.0,     1.0};
        static Color_t LIME         = {205.0/255.0, 220.0/255.0,    57.0/255.0,     1.0};
        static Color_t YELLOW       = {255.0/255.0, 235.0/255.0,    59.0/255.0,     1.0};
        static Color_t AMBER        = {255.0/255.0, 193.0/255.0,    7.0/255.0,      1.0};
        static Color_t ORANGE       = {255.0/255.0, 152.0/255.0,    0.0,            1.0};
        static Color_t DEEP_ORANGE  = {255.0/255.0, 87.0/255.0,     34.0/255.0,     1.0};
        static Color_t BROWN        = {121.0/255.0, 85.0/255.0,     72.0/255.0,     1.0};
        static Color_t GREY         = {158.0/255.0, 158.0/255.0,    158.0/255.0,    1.0};
        static Color_t BLUE_GRAY    = {96.0/255.0,  125.0/255.0,    139.0/255.0,    1.0};
        static Color_t DARK_GRAY    = {33.0/255.0,  33.0/255.0,     33.0/255.0,     1.0};
        static Color_t BLACK        = {0.0,         0.0,            0.0,            1.0};
        static Color_t WHITE        = {250.0/255.0, 250.0/255.0,    250.0/255.0,    1.0};
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
        c.a() = alpha;
        return c;
    }

    }} // namespace name
