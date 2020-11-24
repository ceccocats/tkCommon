
#include <tkCommon/gui/drawables/Drawable.h>
#include <tkCommon/gui/utils/Texture.h>

#include <tkCommon/utils.h>

#include <tkCommon/data/ImageData.h>
#include <tkCommon/data/CalibData.h>

namespace tk{namespace gui{

class CameraAR : public tk::gui::Drawable, public tk::rt::Lockable {

public:

    tk::data::ImageData img;
    tk::gui::Texture<uint8_t>* texture; 
    tk::gui::Texture<uint8_t>* im_texture;

    tk::data::CalibData calib;

    tk::common::Tfpose view;

    std::string name;

    std::vector< tk::gui::Drawable * > drawables;

    int w,h,c;

    float z_near = 1.0, z_far = 10000.0;

    tk::gui::Camera camera;

    bool save = false;

    CameraAR(tk::data::CalibData &calib, int w, int h, std::string name = "AR"){

        this->calib = calib;
        this->name = name;
        this->w = w;
        this->h = h;
        this->c = 4;
        camera.init();
        view = tk::common::Tfpose::Identity();

        float fx = calib.k(0,0);
        float fy = calib.k(1,1);
        float cx = calib.k(0,2);
        float cy = calib.k(1,2);
        
        //camera.projection = glm::perspective<float>(-fov, (float)-w/h, z_near, z_far);

        camera.projection[0][0] = 2.0 * fx / w;
        camera.projection[0][1] = 0.0;
        camera.projection[0][2] = 0.0;
        camera.projection[0][3] = 0.0;

        camera.projection[1][0] = 0.0;
        camera.projection[1][1] = -2.0 * fy / h;
        camera.projection[1][2] = 0.0;
        camera.projection[1][3] = 0.0;

        camera.projection[2][0] = 1.0 - 2.0 * cx / w;
        camera.projection[2][1] = 2.0 * cy / h - 1.0;
        camera.projection[2][2] = (z_far + z_near) / (z_near - z_far);
        camera.projection[2][3] = -1.0;

        camera.projection[3][0] = 0.0;
        camera.projection[3][1] = 0.0;
        camera.projection[3][2] = 2.0 * z_far * z_near / (z_near - z_far);
        camera.projection[3][3] = 0.0;

        img.init();

    }

    void updateImage(tk::data::ImageData &im){
        img = im;
    }

    void onInit(tk::gui::Viewer *viewer){

        texture = new tk::gui::Texture<uint8_t>();
        im_texture = new tk::gui::Texture<uint8_t>();
        texture->init(w, h, c, true);
        im_texture->init(w, h, c, false);

    }

    void imGuiSettings()
    {
        save = save || ImGui::Button("Save Map!");
    }

    void draw(tk::gui::Viewer *viewer) {

        if( img.isChanged() ){
            if(!img.empty()){
                img.lockRead();
                im_texture->setData(img.data);
                img.unlockRead();
            }
            
            glPushMatrix();
            {
                texture->useForRendering();

                glLoadIdentity();

                // Apply projection
                glMultMatrixf(glm::value_ptr(camera.projection));

                // Apply ModelView
                lockWrite();
                camera.modelView = glm::make_mat4((float*)view.matrix().data());
                unlockWrite();
                glm::mat4 mv = glm::make_mat4( (float*)tk::common::odom2tf(0,0,0,M_PI,0,0).matrix().data());
                glm::mat4 m_view = mv*camera.modelView;
                glMultMatrixf(glm::value_ptr(m_view));

                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

                for (auto const& drawable : drawables){
                    if(drawable->enabled){
                        glPushMatrix();
                        glMultMatrixf(drawable->tf.matrix().data());
                        drawable->draw(viewer);
                        glPopMatrix();
                    }
                }

                texture->unuseRendering();

            }
            glPopMatrix();
        }

        

        ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_NoScrollbar);
            
        float imgX = ImGui::GetWindowSize().x-20;
        //int imgY = ImGui::GetWindowSize().y-35;
        //float imgX = textures[i]->width;
        float imgY = imgX / ((float)texture->width / texture->height);
        ImGui::Text("%s",name.c_str());
        auto p = ImGui::GetCursorPos(); 
        ImGui::Image((void*)(intptr_t)im_texture->id(), ImVec2(imgX, imgY));
        ImGui::SetCursorPos(p);
        ImGui::Image((void*)(intptr_t)texture->id(), ImVec2(imgX, imgY));
        ImGui::Separator();

        ImGui::End();
    }

};
}}