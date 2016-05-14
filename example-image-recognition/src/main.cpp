#include "ofMain.h"
#include "ofxDeepBeliefSDK.h"

class ofApp : public ofBaseApp
{
	public:
	ofxDeepBeliefSDK deep;
	ofImage img;

	void setup(){
		deep.load("dog.jpg","../../../networks/jetpac.ntwk");
		img.load("dog.jpg");
	}
	
	void draw(){
		string msg = "";
		vector<string> p = deep.getPrediction();
		for(int i = 0; i < p.size(); i++){
			msg += ofToString(i)+") "+p[i]+"\n";
		}
		img.draw(0,0);
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
