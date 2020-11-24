#pragma once

namespace tk { namespace common {

    template<unsigned N> unsigned force_constexpr_eval() {
        return N;
    };
    unsigned constexpr key(char const *input) {
        return *input ?
            static_cast<unsigned int>(*input) + 33 * key(input + 1) :
            5381;
    }
    template <class T>
    class Map {
    public:
        void clear() {
            _map.clear();
            _mapKeys.clear();
            _keys.clear();
            _vals.clear();
        }
        void add(std::string key) {
            unsigned hash_key = tk::common::key(key.c_str());
            if(!exists(key)) {
                _map[hash_key] = T();
                _keys.push_back(key);
                _vals.push_back(&_map[hash_key]);
                _mapKeys[hash_key] = key;
            } else {
                tkFATAL(key + " key already inside map")
            }
        }
        bool exists(unsigned key) {
            return _map.count(key);
        }
        bool exists(std::string key) {
            unsigned hash_key = tk::common::key(key.c_str());
            return _map.count(hash_key);
        }
        int size() {
            return _map.size();
        }
        T& operator[](unsigned key) {
            tkASSERT(_map.count(key) > 0, std::to_string(key) + " does not exists");
            return _map[key];
        }
        T& operator[](std::string key) {
            unsigned hash_key = tk::common::key(key.c_str());
            tkASSERT(_map.count(hash_key) > 0, key + " does not exists");
            return _map[hash_key];
        }
        const std::vector<T*>& vals() {
            return _vals;
        }
        const std::vector<std::string>& keys() {
            return _keys;
        }
        friend std::ostream& operator<<(std::ostream& os, Map& s) {
            os<<"KeyMap:\n";
            for(int i=0; i<s.size(); i++) {
                os<<i<<": "<<s.keys()[i]<<" "<<s[s.keys()[i]]<<"\n";
            }
            return os;
        }
        Map& operator=(const Map& s)
        {
            _map = s._map;
            _mapKeys = s._mapKeys;
            _keys = s._keys;
            _vals.resize(_map.size());
            // update pointers
            for(int i=0; i<_map.size(); i++) {
                unsigned hash_key = tk::common::key(_keys[i].c_str());
                _vals[i] = &_map[hash_key];
            }
            return *this;
        }
    private: 
        // unordered_map in theory has access time of o(1)
        // we esperienced the std::map has anyway better performance 
        std::map<unsigned, T> _map;
        std::map<unsigned, std::string> _mapKeys;
        std::vector<std::string> _keys;
        std::vector<T*> _vals;

    };

    #define tkKey(A) tk::common::force_constexpr_eval<tk::common::key(A)>()
}}