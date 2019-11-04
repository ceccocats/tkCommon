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

tk::communication::PCAPHandler  handler;
tk::communication::UDPSocket    sender;
std::mutex                      pcapMutex;

tk::gui::Viewer                 *viewer = nullptr;

std::string                     fileLog;
std::string                     filterLog;
int                             port;

bool                            gRun        = true;

int			                    velocity = 1;
int 		                    Npackets, packets;
bool                            running = false;
timeStamp_t                     nulla, prec = 0, now;

tk::data::replayPcap_t          gui_data;


int counterPackets(){

    tk::communication::PCAPHandler counter;
    uint8_t buffer[2000];
    int packets = 0; 
	tkASSERT( counter.initReplay(fileLog,filterLog) );
	while( counter.getPacket(buffer,nulla) != -1)
		packets++;
    counter.close();
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
                usleep(sleep_for*velocity);
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
            gui_data.textOutput = "\nFile:\t\t"+fileLog+"\n\nTime:\t\t" + std::to_string(Npackets*(0.1/156)) + "s\nTotal time:  " + std::to_string(packets*(0.1/156)) + "s";
        }

        gui_data.barNumPacket = Npackets;

		usleep(100);
	}

}



int main(int argc, char *argv[])
{
    tk::exceptions::handleSegfault();

    tk::common::CmdParser   cmd(argv, "Samples for handle ethernet packets");
    fileLog                 = cmd.addOpt("-file", "", "pcap replay file");
    port                    = cmd.addIntOpt("-port", 2368, "pcap replay file");
    cmd.print();

    packets = counterPackets();
    gui_data.barMaxVal = packets;
    gui_data.textOutput = "\nFile:\t\t"+fileLog+"\n\nTime:\t\t" + std::to_string(0) + "s\nTotal time:  " + std::to_string(packets*(0.1/156)) + "s";
	
	viewer = new tk::gui::Viewer();
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

