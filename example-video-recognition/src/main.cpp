#include "ofMain.h"
#include "ofxDeepBeliefSDK.h"

class ofApp : public ofBaseApp
{
	public:
	ofxDeepBeliefSDK deep;
        ofVideoGrabber vid;

	void setup(){
		vid.setDeviceID(0);
	        vid.setDesiredFrameRate(60);
   	        vid.initGrabber(480,320);

		deep.setup("../../../networks/jetpac.ntwk");
	}
	
	void draw(){
		vid.update();
	        string msg;
	        if(vid.isFrameNew()){
                     unsigned char *u = vid.getPixels();
		     deep.setPixels(u,vid.getWidth(),vid.getHeight());
		     vector<string> pred = deep.getPrediction();
		     for(int i = 0; i < pred.size(); i++){
			msg+=pred[i]+"\n";
		     }
                }
		vid.draw(0,0);
		ofDrawBitmapStringHighlight(msg,10,10);
	}

	void keyPressed(int key){
		if(key == 'd')
			deep.printNetwork();
	}
};

int main()
{
    ofSetupOpenGL(1024,768, OF_WINDOW);
    ofRunApp( new ofApp());
}
