#pragma once

namespace tk { namespace communication {

    #pragma pack(push, 1)
    struct AutosarHeader_t {
        uint32_t id;
        uint32_t datalenght;
    };
    #pragma pack(pop)

    class AutosarPkt {

    public:
        uint32_t id;       /**< id of the message */
        uint32_t len;      /**< data length  */
        uint8_t *data;     /**< data pointer */

        uint8_t *buffer;
        uint32_t buflen; /**< len with header */

        /**
         * read an autosar packet header from buffer
         * @param buffer
         * @param buflen
         * @return
         */
        bool read(uint8_t *buffer, int buflen) {

            if (buflen < sizeof(AutosarHeader_t))
                return false;

            AutosarHeader_t header;
            std::memcpy((void*)&header, buffer, sizeof(AutosarHeader_t));

            // update data
            this->id  = __builtin_bswap32(header.id);
            this->len = __builtin_bswap32(header.datalenght);
            this->buffer = buffer;
            this->buflen = sizeof(AutosarHeader_t) + len;
            this->data = buffer + sizeof(AutosarHeader_t);

            //clsMsg("msg id: " + hexId() + " len: " + std::to_string(len) + "\n");
            return buflen <= sizeof(AutosarHeader_t) + header.datalenght;
        }

        /**
         * write an autosar packet header to buffer
         * @param buffer
         * @param buflen
         * @param id
         * @param len
         * @return
         */
        bool write(uint8_t *buffer, int buflen, uint32_t id, uint32_t len) {
            if(len + sizeof(AutosarHeader_t) > buflen)
                return false;

            // update data
            this->id  = id;
            this->len = len;
            this->buffer = buffer;
            this->buflen = sizeof(AutosarHeader_t) + len;
            this->data = buffer + sizeof(AutosarHeader_t);

            AutosarHeader_t header;
            header.id         = __builtin_bswap32(id);
            header.datalenght = __builtin_bswap32(len);
            std::memcpy(buffer, (void*)&header, sizeof(AutosarHeader_t));
            return true;
        }

        std::string hexId() {
            std::stringstream stream;
            stream << std::hex << id;
            return stream.str();
        }
    };


}}