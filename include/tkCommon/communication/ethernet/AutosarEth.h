#pragma once

namespace tk { namespace communication {

    #pragma pack(push, 1)
    typedef struct AutosarHeader_t {
        uint32_t id;
        uint32_t datalenght;
    };
    #pragma pack(pop)

    class AutosarPkt {

    public:
        uint32_t id;       /**< id of the message */
        uint32_t len;      /**< data length  */
        uint8_t *data;     /**< data pointer */

        uint64_t totalLen; /**< len with header */

        bool init(uint8_t *buffer, int buflen) {

            if (buflen < sizeof(AutosarHeader_t))
                return false;

            AutosarHeader_t header;
            std::memcpy((void*)&header, buffer, sizeof(AutosarHeader_t));

            // update data
            id  = __builtin_bswap32(header.id);
            len = __builtin_bswap32(header.datalenght);
            totalLen = sizeof(AutosarHeader_t) + len;
            data = buffer + sizeof(AutosarHeader_t);

            //clsMsg("msg id: " + hexId() + " len: " + std::to_string(len) + "\n");
            return buflen <= sizeof(AutosarHeader_t) + header.datalenght;
        }

        std::string hexId() {
            std::stringstream stream;
            stream << std::hex << id;
            return stream.str();
        }
    };


}}