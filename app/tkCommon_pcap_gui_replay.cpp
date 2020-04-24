#include <string>
#include <cstdlib>
#include <iostream>
#include <mutex>

//viewer
#include <tkCommon/gui/Viewer.h>

//communication
#include <tkCommon/communication/ethernet/PCAPHandler.h>
#include <tkCommon/communication/ethernet/UDPSocket.h>

#include <tkCommon/data/LidarData.h>

#include <tkCommon/exceptions.h>
#include <tkCommon/terminalFormat.h>

struct replayPcap_t{

        float           velocity      = 1;
        bool            pressedStart  = false;
        bool            pressedStop   = false;
        bool            pressedBar    = false;
        int             barNumPacket  = 0;
        int             barMinVal     = 0;
        int             barMaxVal     = 0;
        std::string     textOutput    = ""; 
};

class MyViewer : public tk::gui::Viewer {
    private:
        //gui replay
        replayPcap_t *replaypcap = nullptr;

    public:
        MyViewer() {}
        ~MyViewer() {}

        void init() {
            tk::gui::Viewer::init();
        }

        void draw() {
            tk::gui::Viewer::draw();

            if(replaypcap == nullptr)
                return;

            bool a = true;
            ImGui::Begin("pktseeker",&a, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove);
            ImGui::SetWindowSize(ImVec2(400,200));
            ImGui::SetWindowPos(ImVec2(0,0));
            replaypcap->pressedBar = ImGui::SliderInt("Packet", &replaypcap->barNumPacket, replaypcap->barMinVal, replaypcap->barMaxVal);
            ImGui::InputFloat("Frame ratio", &replaypcap->velocity);
            replaypcap->pressedStart = ImGui::Button("Start");
            ImGui::SameLine();
            replaypcap->pressedStop = ImGui::Button("Stop");
            ImGui::Text("%s", replaypcap->textOutput.c_str());
            ImGui::End();
        }
        void setGuiReplay(replayPcap_t *replay){this->replaypcap = replay;}
};

tk::communication::PCAPHandler  handler;
tk::communication::UDPSocket    sender;
std::mutex                      pcapMutex;

MyViewer                        *viewer = nullptr;

std::string                     fileLog;
std::string                     filterLog;
int                             port;

bool                            gRun        = true;

int			                    velocity = 1;
int 		                    Npackets, packets;
bool                            running = false;
timeStamp_t                     nulla, prec = 0, now, start, end;
double                          timeFor1Pkt;
replayPcap_t          gui_data;


int counterPackets(){

    tk::communication::PCAPHandler counter;
    uint8_t buffer[2000];
    int packets = 0; 
	tkASSERT( counter.initReplay(fileLog,filterLog) );
    counter.getPacket(buffer,start);
	while( counter.getPacket(buffer,end) != -1)
		packets++;
    counter.close();
    timeFor1Pkt = ( (double)(end-start)/1000000.0 ) / packets;
	return packets;

}

void initPcap(int n)
{
    uint8_t buffer[2000];

	pcapMutex.lock();

	if(n < Npackets){

		handler.close();
        tkASSERT( handler.initReplay(fileLog,filterLog) );

		for(int d=0; d < n; d++)
			handler.getPacket(buffer,nulla);

	}else{

		for(int d=Npackets; d < n; d++)
			handler.getPacket(buffer,nulla);
	}

    Npackets = n;

	pcapMutex.unlock();

}


void* replayLoop(void*){

    tkASSERT( handler.initReplay(fileLog,filterLog) );
    tkASSERT( sender.initSender(port,"127.0.0.1") );
    uint8_t buffer[2000];
    

    while(gRun){      

        if(running == false){
            usleep(1000);
            continue;
        }

        pcapMutex.lock();
        int n = handler.getPacket(buffer,now);
        pcapMutex.unlock();

        if(n == -1){
            running = false;
            gui_data.textOutput = "\nEnd of packet file. Press start.";
            pcapMutex.unlock();
            initPcap(0);
            continue;
        }

        sender.send(buffer,n);

        if(prec != 0){

            if(now > prec){
                
                timeStamp_t sleep_for = (now-prec);
                usleep(sleep_for/velocity);
            }
        }

        prec = now;

        Npackets++;
    }

    handler.close();
    sender.close();
}

void* control(void*){

	while(gRun){

		if(velocity != gui_data.velocity && gui_data.velocity > 0){
			velocity = gui_data.velocity;
		}

		if(gui_data.pressedBar){
			initPcap(gui_data.barNumPacket);
			prec = 0;
		}

		if(gui_data.pressedStart){
			running = true;
			prec = 0;
		}

		if(gui_data.pressedStop){
			running = false;
		}

        if (running && (Npackets % 100) == 0)
        {
            gui_data.textOutput = "\nFile:\t\t"+fileLog+"\n\nTime:\t\t" + std::to_string(Npackets*timeFor1Pkt) + "s\nTotal time:  " + std::to_string(packets*(0.1/156)) + "s";
        }

        gui_data.barNumPacket = Npackets;

		usleep(100);
	}

}



int main(int argc, char *argv[])
{
    tk::exceptions::handleSegfault();

    tk::common::CmdParser   cmd(argv, "Samples for handle ethernet packets");
    fileLog                 = cmd.addArg("file", "", "pcap replay file");
    port                    = cmd.addIntOpt("-port", 2368, "pcap replay file");
    cmd.parse();

    if(fileLog == ""){
        return 0;
    }

    packets = counterPackets();
    gui_data.barMaxVal = packets;
    gui_data.textOutput = "\nFile:\t\t"+fileLog+"\n\nTime:\t\t" + std::to_string(0) + "s\nTotal time:  " + std::to_string((double)(end-start)/1000000.0) + "s";
	
	viewer = new MyViewer();
    viewer->setWindowName("Lidar replay control panel");
	viewer->width = 400;
	viewer->height = 200;
    viewer->setGuiReplay(&gui_data);
    viewer->init();

	pthread_t t1;
	pthread_create(&t1, NULL, control, NULL);
    pthread_create(&t1, NULL, replayLoop, NULL);

    // start APP
    viewer->run();

	return 0;
}

