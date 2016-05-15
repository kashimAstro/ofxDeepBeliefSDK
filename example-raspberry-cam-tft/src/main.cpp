#include "ofMain.h"
#include "ofxDeepBeliefSDK.h"
#include "ofxRPiCameraVideoGrabber.h"

class ofApp : public ofBaseApp
{
	public:
	ofxDeepBeliefSDK deep;
        ofxRPiCameraVideoGrabber videoGrabber;
        SessionConfig sessionConfig;

	void setup(){
	        sessionConfig.width = 320;
	        sessionConfig.height = 480;
	        sessionConfig.framerate = 30;
	        sessionConfig.isUsingTexture(); 
                sessionConfig.enablePixels = true;
	        videoGrabber.setup(sessionConfig);

		deep.setup("../../../networks/jetpac.ntwk");
	}
	
	void draw(){
	        string msg;
	     	unsigned char *u = videoGrabber.getPixels();
		deep.setPixels(u,videoGrabber.getWidth(),videoGrabber.getHeight());
	     	vector<string> pred = deep.getPrediction();
	     	for(int i = 0; i < pred.size(); i++){
			msg+=pred[i]+"\n";
	     	}
	        videoGrabber.draw();
		ofDrawBitmapStringHighlight(msg,10,10);
	}

	void keyPressed(int key){
		if(key == 'd')
			deep.printNetwork();
	}
};

int main()
{
    ofSetupOpenGL(320,480, OF_WINDOW);
    ofRunApp( new ofApp());
}
