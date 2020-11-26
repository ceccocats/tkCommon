
#include "tkCommon/gui/drawables/CameraAR.h"

tk::gui::CameraAR::CameraAR(tk::data::CalibData &calib, int channels, std::string name){

    this->calib = calib;
    this->name = name;
    this->w = calib.w;
    this->h = calib.h;
    this->c = channels;
    camera.init();
    view = tk::common::Tfpose::Identity();

    float fx = calib.k.data_h[0*3+0];
    float fy = calib.k.data_h[1*3+1];
    float cx = calib.k.data_h[0*3+2];
    float cy = calib.h - calib.k.data_h[1*3+2];
    

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

    //camera.projection = glm::perspective<float>(-60, (float)-w/h, z_near, z_far);
    img.init();

    counter = 0;
}

void 
tk::gui::CameraAR::updateImage(tk::data::ImageData &im){
    img = im;
}

void 
tk::gui::CameraAR::onInit(tk::gui::Viewer *viewer){

    texture = new tk::gui::Texture<uint8_t>();
    im_texture = new tk::gui::Texture<uint8_t>();
    texture->init(w, h, 4, true);
    im_texture->init(w, h, c, false);

}

void 
tk::gui::CameraAR::imGuiSettings()    {
    save = save || ImGui::Button("Save Map!");
}

void 
tk::gui::CameraAR::draw(tk::gui::Viewer *viewer) {

    if( img.isChanged(counter) ){
        if(!img.empty()){
            img.lockRead();
            im_texture->setData(img.data);
            img.unlockRead();
        }        
    }    
    glPushMatrix();
    {
        texture->useForRendering();

        glLoadIdentity();

        // Apply projection
        glMultMatrixf(glm::value_ptr(camera.projection));

        // Apply ModelView
        lockWrite();
        tk::common::Tfpose fix_axis = tk::common::odom2tf(0,0,0,M_PI/2,0, -M_PI/2);
        tk::common::Tfpose view_fixed = (view*fix_axis).inverse();

        //camera.modelView = viewer->camera.modelView; 
        camera.modelView = glm::make_mat4((float*)view_fixed.matrix().data());
        unlockWrite();

        glm::mat4 m_view = camera.modelView;
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

