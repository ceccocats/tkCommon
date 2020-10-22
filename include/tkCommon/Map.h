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
        }
        void add(std::string key) {
            unsigned hash_key = tk::common::key(key.c_str());
            if(!exists(key)) {
                _keys.push_back(key);
            } else 
            _mapKeys[hash_key] = key;
            _map[hash_key] = T();
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
        const  std::vector<std::string>& keys() {
            return _keys;
        }
        friend std::ostream& operator<<(std::ostream& os, Map& s) {
            for(int i=0; i<s.size(); i++) {
                os<<s.keys()[i]<<" "<<s[s.keys()[i]]<<"\n";
            }
            return os;
        }
    private: 
        // unordered_map in theory has access time of o(1)
        // we esperienced the std::map has anyway better performance 
        std::map<unsigned, T> _map;
        std::map<unsigned, std::string> _mapKeys;
        std::vector<std::string> _keys;
    };

    #define tkKey(A) tk::common::force_constexpr_eval<tk::common::key(A)>()
}}