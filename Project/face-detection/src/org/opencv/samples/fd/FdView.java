/**
* Copyright (C) 2013 Imran Akthar (www.imranakthar.com)
* imran@imranakthar.com
*/
package org.opencv.samples.fd;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.SurfaceHolder;

class FdView extends SampleCvViewBase {
    private static final String   TAG = "Sample::FdView";
    private Mat                   mRgba;
    private Mat                   mGray;
    private File                  mCascadeFile;
    private CascadeClassifier     mJavaDetector;
    private DetectionBasedTracker mNativeDetector;

    private static final Scalar   FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    
    public static final int       JAVA_DETECTOR     = 0;
    public static final int       NATIVE_DETECTOR   = 1;
    
    private int                   mDetectorType     = JAVA_DETECTOR;

    private float                 mRelativeFaceSize = 0;
    private int					  mAbsoluteFaceSize = 0;
    private Mat mLogoMilka4;
    private static final int BOUNDARY = 35;   
    private Mat mIntermediateMat;
    enum HeadPoseStatus {INIT,TAKESNAP,TRACKING};
   
    ///////////////////DETECTORS
    //FeatureDetector siftDetector;// = FeatureDetector.create(FeatureDetector.SIFT);
    //FeatureDetector surfDetector;// = FeatureDetector.create(FeatureDetector.SURF);
    FeatureDetector fastDetector;// = FeatureDetector.create(FeatureDetector.FAST);
    FeatureDetector orbDetector;// = FeatureDetector.create(FeatureDetector.ORB);
    
    ///////////////////DESCRIPTORS
    //DescriptorExtractor siftDescriptor;// = DescriptorExtractor.create(DescriptorExtractor.SIFT);
    //DescriptorExtractor surfDescriptor;// = DescriptorExtractor.create(DescriptorExtractor.SURF);
    DescriptorExtractor briefDescriptor;// = DescriptorExtractor.create(DescriptorExtractor.BRIEF);
    DescriptorExtractor flannDescriptor;// = DescriptorExtractor.create(DescriptorExtractor.BRIEF);
    DescriptorExtractor orbDescriptor;// = DescriptorExtractor.create(DescriptorExtractor.ORB);
    ///////////////////DATABASE
    //Vector<KeyPoint> vectorMilka1 = new Vector<KeyPoint>();
    //Vector<KeyPoint> vectorMilka2 = new Vector<KeyPoint>();
    //Vector<KeyPoint> vectorMilka3 = new Vector<KeyPoint>();
    //Vector<KeyPoint> vectorMilka4 = new Vector<KeyPoint>();
    MatOfKeyPoint vectorMilka4;// = new MatOfKeyPoint();
    //Mat descriptorMilka1 = new Mat();
    //Mat descriptorMilka2 = new Mat();
    //Mat descriptorMilka3 = new Mat(); 
    Mat descriptorMilka4,homography;// = new Mat();
    ///////////////////VIDEO
    //Vector<KeyPoint> vectorFrame = new Vector<KeyPoint>();
    MatOfKeyPoint vectorFrame;// = new MatOfKeyPoint();
    Mat descriptorFrame;// = new Mat();

    DescriptorMatcher matcherHamming;// = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);
    DescriptorMatcher matcherFlann;// = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);
    DescriptorMatcher matcherBruteForce;// = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_SL2);
    //Vector<DMatch> matches = new Vector<DMatch>();
    MatOfDMatch matches;// = new MatOfDMatch();    
    //Vector<Mat> siftDescriptors;// = new Vector<Mat>();
    //Vector<Mat> surfDescriptors;// = new Vector<Mat>();
    Vector<Mat> briefDescriptors;// = new Vector<Mat>();
    Vector<Mat> orbDescriptors;// = new Vector<Mat>();
    Vector<Mat> flannDescriptors;// = new Vector<Mat>();
    enum viewMode {VIEW_MODE_SIFT,VIEW_MODE_SURF,VIEW_MODE_BRIEF,VIEW_MODE_ORB,VIEW_MODE_TEST};
    viewMode VM;
    public boolean takesnap;//=false;
    public boolean TRACKING = false;
    Mat logotemp;
    HeadPoseStatus hp;
    
    public void setMinFaceSize(float faceSize)
    {
		mRelativeFaceSize = faceSize;
		mAbsoluteFaceSize = 0;
    }
    
    public void setDetectorType(int type)
    {
    	if (mDetectorType != type)
    	{
    		mDetectorType = type;
    		
    		if (type == NATIVE_DETECTOR)
    		{
    			Log.i(TAG, "Detection Based Tracker enabled");
    			mNativeDetector.start();
    		}
    		else
    		{
    			Log.i(TAG, "Cascade detector enabled");
    			mNativeDetector.stop();
    		}
    	}
    }

    public FdView(Context context) {
        super(context);
       
        orbDetector = FeatureDetector.create(FeatureDetector.ORB);
        fastDetector = FeatureDetector.create(FeatureDetector.FAST);
        
        orbDescriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        briefDescriptor = DescriptorExtractor.create(DescriptorExtractor.BRIEF);
        
        vectorMilka4 = new MatOfKeyPoint();
     
        descriptorMilka4 = new Mat();
       
        
        vectorFrame = new MatOfKeyPoint();
    
        
        descriptorFrame = new Mat();
      
        
        matcherHamming = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);
        matcherFlann = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        
        matcherBruteForce = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_SL2);
       
                
        matches = new MatOfDMatch();
        
        
        orbDescriptors = new Vector<Mat>();
        briefDescriptors = new Vector<Mat>();
        flannDescriptors=new Vector<Mat>();
        takesnap=false;
        hp=HeadPoseStatus.INIT;
      
        
        VM =viewMode.VIEW_MODE_TEST;
      
        try {
        	//integrating both under modish project
        	InputStream is = context.getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if (mJavaDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                mJavaDetector = null;
            } else
                Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

            mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);
            
            cascadeDir.delete();

        } catch (IOException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
        }
    }
    //public void fillDB(Mat mLogo,Vector<KeyPoint> vector,Mat descriptor){
    public void fillDBORB(Mat mLogo,MatOfKeyPoint vector,Mat descriptor)
    {
    	//ORB 
        orbDetector.detect( mLogo, vector );
        orbDescriptor.compute(mLogo, vector, descriptor);
        orbDescriptors.add(descriptor);
    	
       
      
      }
    
    public void fillDBBRIEF(Mat mLogo,MatOfKeyPoint vector,Mat descriptor)
    {
    	
    	
        //FAST+BRIEF 
        fastDetector.detect( mLogo, vector );
        briefDescriptor.compute(mLogo, vector, descriptor);
        briefDescriptors.add(descriptor);
      
      }
    public void fillDFLANN(Mat mLogo,MatOfKeyPoint vector,Mat descriptor)
    {
    	
    	
        //FAST+BRIEF 
        fastDetector.detect( mLogo, vector );
        flannDescriptor.compute(mLogo, vector, descriptor);
        flannDescriptors.add(descriptor);
      
      }
    public Rect[] FaceDetect(Mat mRgba,Mat mGray )
    {
    	if (mAbsoluteFaceSize == 0)
        {
        	int height = mGray.rows();
        	if (Math.round(height * mRelativeFaceSize) > 0);
        	{
        		mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
        	}
        	mNativeDetector.setMinFaceSize(mAbsoluteFaceSize);
        }
        
        MatOfRect faces = new MatOfRect();
        
        if (mDetectorType == JAVA_DETECTOR)
        {
        	if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2, 2 // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        , new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else if (mDetectorType == NATIVE_DETECTOR)
        {
        	if (mNativeDetector != null)
        		mNativeDetector.detect(mGray, faces);
        }
        else
        {
        	Log.e(TAG, "Detection method is not selected!");
        }
        
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 3);
      
        //facearray1=FaceDetect(mRgba,mGray);
       
    	return facesArray;
    	
    }
    
    
    @Override
	public void surfaceCreated(SurfaceHolder holder) {
        synchronized (this) {
            // initialize Mats before usage
            mGray = new Mat();
           
            mRgba = new Mat();
           
            mIntermediateMat = new Mat();
            matches = new MatOfDMatch();
            vectorFrame = new MatOfKeyPoint();
            descriptorFrame = new Mat(); 
            
            
        }

        super.surfaceCreated(holder);
	}

	@Override
    protected Bitmap processFrame(VideoCapture capture) {
		 
		//Mat logotemp = null;
 	  //imran FLANN BAsed https://groups.google.com/forum/#!msg/android-opencv/BdfLb78lEh8/T3FK7BXTajsJ
		  //capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
   		//capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
   	     // capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
		
		switch (VM) {
	        case VIEW_MODE_SIFT:
	         
	            break;
	        case VIEW_MODE_SURF:
	           
	            break;
	            
	        case VIEW_MODE_TEST:
	        			
	        		   //capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
	        			capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
	        			//capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
	        			capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
		        	   if(hp==HeadPoseStatus.INIT)
		         	   {
		        		
		         		Rect[] facearray= FaceDetect(mRgba,mGray );
		         		
		         		for (int i = 0; i < facearray.length; i++)
		                     Core.rectangle(mRgba, facearray[i].tl(), facearray[i].br(), FACE_RECT_COLOR, 3);   
		         		
		        		 		if(facearray.length>0)
		        		 		{
		        		 		//logotemp = new Mat(facearray[0].size(),CvType.CV_8UC4);
		        		 		//Rect roi = new Rect((int)facearray[0].tl().x,(int)(facearray[0].tl().y),facearray[0].width,(int)(facearray[0].height));
		        		 		//logotemp=mRgba.submat(roi);
		        		 		
		        		 		mLogoMilka4 = new Mat(facearray[0].size(),CvType.CV_8UC4);
		        		 		Rect roi = new Rect((int)facearray[0].tl().x,(int)(facearray[0].tl().y),facearray[0].width,(int)(facearray[0].height));
		        		 		mLogoMilka4=mRgba.submat(roi);
		        		 		}
				         		
		         	   }
		         	   if (hp==HeadPoseStatus.TAKESNAP)
		                 {
		                    // if(mLogoMilka4==null)
		         			// mLogoMilka4 = new Mat();
		                    // mLogoMilka4=logotemp.clone();
		                    // fillDBORB(mLogoMilka4,vectorMilka4,descriptorMilka4);
		                    // hp=HeadPoseStatus.TRACKING;
		                    
		         		   
		         		   //if(mLogoMilka4==null)
			         			// mLogoMilka4 = new Mat();
			                   //  mLogoMilka4=logotemp.clone();
			                     fillDBORB(mLogoMilka4,vectorMilka4,descriptorMilka4);
			                     hp=HeadPoseStatus.TRACKING;
		                    
		                 }
		        		
		        	   if(hp==HeadPoseStatus.TRACKING)
		                {
		        	   	matcherHamming = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);
		        	   	matcherHamming.add(orbDescriptors);
		                matcherHamming.train();// proba
		               
		                capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
		                Imgproc.resize(mGray, mGray,new Size(480,320)); 
		                orbDetector.detect( mGray, vectorFrame );
		                orbDescriptor.compute(mGray, vectorFrame, descriptorFrame);
		                matcherHamming.match(descriptorFrame, matches); 
		             
		               ///imran taking code from here
		                //http://stackoverflow.com/questions/12783159/filtering-matofdmatch
		                List<DMatch> matchesList = matches.toList();
		                double maxDistance = 0;
		                double minDistance = 100;

		                int rowCount = matchesList.size();
		                for (int i = 0; i < rowCount; i++) {
		                    double dist = matchesList.get(i).distance;
		                    if (dist < minDistance) minDistance = dist;
		                    if (dist > maxDistance) maxDistance = dist;
		                }

		                List<DMatch> goodMatchesList = new ArrayList<DMatch>();
		                double upperBound = 3 * minDistance;
		                while(goodMatchesList.size()<100)
		                {
		                for (int i = 0; i < rowCount; i++)
		                {
		                    if (matchesList.get(i).distance < upperBound) 
		                    {
		                        goodMatchesList.add(matchesList.get(i));
		                    }
		                }
		                }
		                
		                
		                
		        MatOfDMatch matchesXXX = new MatOfDMatch();
		        matchesXXX.fromList(goodMatchesList); 
                Mat nGray = new Mat();
                Mat nLogo = new Mat();
                Mat nRgba = new Mat();
                Imgproc.cvtColor(mGray, nGray, Imgproc.COLOR_RGBA2RGB, 3);
                Imgproc.cvtColor(mLogoMilka4, nLogo, Imgproc.COLOR_RGBA2BGR, 3);
                Features2d.drawMatches(nGray, vectorFrame, nLogo, vectorMilka4, matchesXXX, nRgba);
               // Features2d.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags)    .drawMatches(nGray, vectorFrame, nLogo, vectorMilka4, matchesXXX, nRgba);
                Imgproc.cvtColor(nRgba, mRgba, Imgproc.COLOR_RGB2RGBA, 4);
                //working upto here
                //imran ..taking from tutorial.
                              
                MatOfPoint2f obj=new MatOfPoint2f();
                List<Point> objtemp = new ArrayList<Point>();
                MatOfPoint2f scene=new MatOfPoint2f();
                List<Point> scenetemp = new ArrayList<Point>();
                Point a=new Point();
                Point b=new Point();
                //imran taking from opencv tutorial
                        
                for( int i = 0; i < matchesXXX.toList().size() ; i++ )
                {
                int j=matchesXXX.toList().get(i).queryIdx;
                if(j<vectorFrame.toList().size() && j<vectorMilka4.toList().size())
                {
                a= vectorFrame.toList().get(j).pt;
                objtemp.add(a);
                b= vectorMilka4.toList().get(j).pt;
                scenetemp.add(b);
                }
                }
                obj.fromList(objtemp);
                scene.fromList(scenetemp);
                
                Log.i("FdView","before calling homography= Calib3d.findHomography( obj, scene, Calib3d.RANSAC,0.5);");
                homography= Calib3d.findHomography( obj, scene, Calib3d.RANSAC,0.5);
                Log.i("FdView","after calling homography= Calib3d.findHomography( obj, scene, Calib3d.RANSAC,0.5);");
                
                
                MatOfPoint2f obj_corners=new MatOfPoint2f();
                List<Point> obj_cornerstemp = new ArrayList<Point>();
              
                obj_cornerstemp.add(new Point(0,0)); 
                obj_cornerstemp.add(new Point(nLogo.cols(),0));
                obj_cornerstemp.add(new Point(nLogo.cols(),nLogo.rows()));
                obj_cornerstemp.add(new Point(0,nLogo.rows()));
                obj_corners.fromList(obj_cornerstemp);
                //imran again taking inspiration from first for loop
                              
                MatOfPoint2f scene_corners=new MatOfPoint2f();
                //imran http://stackoverflow.com/questions/9321307/image-perspective-transform-using-android-opencv
                Log.i("FdView","before calling Core.perspectiveTransform( obj_corners, scene_corners,homography);");
                Core.perspectiveTransform( obj_corners, scene_corners,homography);
                Log.i("FdView","after calling Core.perspectiveTransform( obj_corners, scene_corners,homography);");
                MatOfPoint2f offset=new MatOfPoint2f();
                List<Point> offsettemp = new ArrayList<Point>();
                offsettemp.add(new Point(nLogo.cols(),0));
                offset.fromList(offsettemp);
                
                Log.i("scene_corners","0"+scene_corners.toList().get(0).toString());
                Log.i("scene_corners","1"+scene_corners.toList().get(1).toString());
                Log.i("scene_corners","2"+scene_corners.toList().get(2).toString());
                Log.i("scene_corners","3"+scene_corners.toList().get(3).toString());
                Point Point0 =new Point();
                Point0=scene_corners.toList().get(0);
                Point0.x=Point0.x+nLogo.cols();
                Point Point1 =new Point();
                Point1=scene_corners.toList().get(1);
                Point1.x=Point1.x+nLogo.cols();
                Point Point2 =new Point();
                Point2=scene_corners.toList().get(2);
                Point2.x=Point2.x+nLogo.cols();
                Point Point3 =new Point();
                Point3=scene_corners.toList().get(3);
                Point3.x=Point3.x+nLogo.cols();
                
                
               //imran issue similar http://answers.opencv.org/question/983/object-detection-using-surf-flann/
                /*Core.line(mRgba,scene_corners.toList().get(0),scene_corners.toList().get(1),new Scalar(255,0, 0), 10 );
                Core.line(mRgba,scene_corners.toList().get(1),scene_corners.toList().get(2),new Scalar(255,0, 0), 10 );
                Core.line(mRgba,scene_corners.toList().get(2),scene_corners.toList().get(3),new Scalar(255,0, 0), 10 );
                Core.line(mRgba,scene_corners.toList().get(3),scene_corners.toList().get(0),new Scalar(255,0, 0), 10 );*/
                Core.line(mRgba,Point0,Point1,new Scalar(255,0, 0), 10 );
                Core.line(mRgba,Point1,Point2,new Scalar(255,0, 0), 10 );
                Core.line(mRgba,Point2,Point3,new Scalar(255,0, 0), 10 );
                Core.line(mRgba,Point3,Point0,new Scalar(255,0, 0), 10 );
		              
		                }
		            break;  

	            
	        case VIEW_MODE_BRIEF:
	        	
	        	  if(hp==HeadPoseStatus.INIT)
	         	   {
	         		
	         		capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
	         	    capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
	         		Rect[] facearray= FaceDetect(mRgba,mGray );
	         		
	         		for (int i = 0; i < facearray.length; i++)
	                     Core.rectangle(mRgba, facearray[i].tl(), facearray[i].br(), FACE_RECT_COLOR, 3);   
	         		
	        		 		if(facearray.length>0)
	        		 		{
	        		 		logotemp = new Mat(facearray[0].size(),CvType.CV_8UC4);
	        		 		Rect roi = new Rect((int)facearray[0].tl().x,(int)(facearray[0].tl().y),facearray[0].width,(int)(facearray[0].height));
	        		 		logotemp=mRgba.submat(roi);
	        		 		}
	         		
	         	   }
	         	   if (hp==HeadPoseStatus.TAKESNAP)
	                 {
	                     if(mLogoMilka4==null)
	         			 mLogoMilka4 = new Mat();
	                     mLogoMilka4=logotemp.clone();
	                     fillDBBRIEF(mLogoMilka4,vectorMilka4,descriptorMilka4);
	                     hp=HeadPoseStatus.TRACKING;
	                    
	                 }
	        		
	        	   if(hp==HeadPoseStatus.TRACKING)
	                {
        		   matcherHamming = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);
                   matcherHamming.add(briefDescriptors);
                   matcherHamming.train();// proba

                   capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
                   Imgproc.resize(mGray, mGray,new Size(480,320)); 
                   fastDetector.detect( mGray, vectorFrame );
                   briefDescriptor.compute(mGray, vectorFrame, descriptorFrame);

                   matcherHamming.match(descriptorFrame, matches); 
	             
	               ///imran taking code from here
	                //http://stackoverflow.com/questions/12783159/filtering-matofdmatch
	                List<DMatch> matchesList = matches.toList();
	                double maxDistance = 0;
	                double minDistance = 100;

	                int rowCount = matchesList.size();
	                for (int i = 0; i < rowCount; i++) {
	                    double dist = matchesList.get(i).distance;
	                    if (dist < minDistance) minDistance = dist;
	                    if (dist > maxDistance) maxDistance = dist;
	                }

	                List<DMatch> goodMatchesList = new ArrayList<DMatch>();
	                double upperBound = 3 * minDistance;
	                while(goodMatchesList.size()<100)
	                {
	                for (int i = 0; i < rowCount; i++)
	                {
	                    if (matchesList.get(i).distance < upperBound) 
	                    {
	                        goodMatchesList.add(matchesList.get(i));
	                    }
	                }
	                }
	                
	                
	                
	        MatOfDMatch matchesXXX = new MatOfDMatch();
	        matchesXXX.fromList(goodMatchesList); 
           Mat nGray = new Mat();
           Mat nLogo = new Mat();
           Mat nRgba = new Mat();
           Imgproc.cvtColor(mGray, nGray, Imgproc.COLOR_RGBA2RGB, 3);
           Imgproc.cvtColor(mLogoMilka4, nLogo, Imgproc.COLOR_RGBA2BGR, 3);
           Features2d.drawMatches(nGray, vectorFrame, nLogo, vectorMilka4, matchesXXX, nRgba);
          // Features2d.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg, matchColor, singlePointColor, matchesMask, flags)    .drawMatches(nGray, vectorFrame, nLogo, vectorMilka4, matchesXXX, nRgba);
           Imgproc.cvtColor(nRgba, mRgba, Imgproc.COLOR_RGB2RGBA, 4);
           //working upto here
           //imran ..taking from tutorial.
                         
           MatOfPoint2f obj=new MatOfPoint2f();
           List<Point> objtemp = new ArrayList<Point>();
           MatOfPoint2f scene=new MatOfPoint2f();
           List<Point> scenetemp = new ArrayList<Point>();
           Point a=new Point();
           Point b=new Point();
           //imran taking from opencv tutorial
                   
           for( int i = 0; i < matchesXXX.toList().size() ; i++ )
           {
           int j=matchesXXX.toList().get(i).queryIdx;
           if(j<vectorFrame.toList().size() && j<vectorMilka4.toList().size())
           {
           a= vectorFrame.toList().get(j).pt;
           objtemp.add(a);
           b= vectorMilka4.toList().get(j).pt;
           scenetemp.add(b);
           }
           }
           obj.fromList(objtemp);
           scene.fromList(scenetemp);
           homography= Calib3d.findHomography( obj, scene, Calib3d.RANSAC,10);
           Log.i("homography matrix","homography"+homography.toString());
           //homography= Calib3d.findHomography( obj, scene);
           MatOfPoint2f obj_corners=new MatOfPoint2f();
           List<Point> obj_cornerstemp = new ArrayList<Point>();
         
           obj_cornerstemp.add(new Point(0,0)); 
           obj_cornerstemp.add(new Point(nLogo.cols(),0));
           obj_cornerstemp.add(new Point(nLogo.cols(),nLogo.rows()));
           obj_cornerstemp.add(new Point(0,nLogo.rows()));
           obj_corners.fromList(obj_cornerstemp);
           //imran again taking inspiration from first for loop
                         
           MatOfPoint2f scene_corners=new MatOfPoint2f();
           //imran http://stackoverflow.com/questions/9321307/image-perspective-transform-using-android-opencv
           Core.perspectiveTransform( obj_corners, scene_corners,homography);
           MatOfPoint2f offset=new MatOfPoint2f();
           List<Point> offsettemp = new ArrayList<Point>();
           offsettemp.add(new Point(nLogo.cols(),0));
           offset.fromList(offsettemp);
           
           
           Point Point0 =new Point();
           Point0=scene_corners.toList().get(0);
           Point0.x=Point0.x+nLogo.cols();
           Point Point1 =new Point();
           Point1=scene_corners.toList().get(1);
           Point1.x=Point1.x+nLogo.cols();
           Point Point2 =new Point();
           Point2=scene_corners.toList().get(2);
           Point2.x=Point2.x+nLogo.cols();
           Point Point3 =new Point();
           Point3=scene_corners.toList().get(3);
           Point3.x=Point3.x+nLogo.cols();
           Log.i("scene_corners","0"+Point0);
           Log.i("scene_corners","1"+Point1);
           Log.i("scene_corners","2"+Point2);
           Log.i("scene_corners","3"+Point3);
           
           
          //imran issue similar http://answers.opencv.org/question/983/object-detection-using-surf-flann/
           /*Core.line(mRgba,scene_corners.toList().get(0),scene_corners.toList().get(1),new Scalar(255,0, 0), 10 );
           Core.line(mRgba,scene_corners.toList().get(1),scene_corners.toList().get(2),new Scalar(255,0, 0), 10 );
           Core.line(mRgba,scene_corners.toList().get(2),scene_corners.toList().get(3),new Scalar(255,0, 0), 10 );
           Core.line(mRgba,scene_corners.toList().get(3),scene_corners.toList().get(0),new Scalar(255,0, 0), 10 );*/
           Core.line(mRgba,Point0,Point1,new Scalar(255,0, 0), 10 );
           Core.line(mRgba,Point1,Point2,new Scalar(255,0, 0), 10 );
           Core.line(mRgba,Point2,Point3,new Scalar(255,0, 0), 10 );
           Core.line(mRgba,Point3,Point0,new Scalar(255,0, 0), 10 );
	              
	                }
	            break;  

	        case VIEW_MODE_ORB:
	            //TODO ORB
	           try{
	        	   if(hp==HeadPoseStatus.INIT)
	         	   {
	         		
	         		capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
	         	    capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
	         		Rect[] facearray= FaceDetect(mRgba,mGray );
	         		
	         		for (int i = 0; i < facearray.length; i++)
	                     Core.rectangle(mRgba, facearray[i].tl(), facearray[i].br(), FACE_RECT_COLOR, 3);   
	         		
	        		 		if(facearray.length>0)
	        		 		{
	        		 		logotemp = new Mat(facearray[0].size(),CvType.CV_8UC4);
	        		 		Rect roi = new Rect((int)facearray[0].tl().x,(int)(facearray[0].tl().y),facearray[0].width,(int)(facearray[0].height));
	        		 		logotemp=mRgba.submat(roi);
	        		 		}
	         		//taking snap 
	        	   Log.i("ProcessFrane","in Takesnap Loop ");
	        	   //Log.i("ProcessFrane", );
	         	   }
	         	   if (hp==HeadPoseStatus.TAKESNAP)
	                 {
	                     if(mLogoMilka4==null)
	         			 mLogoMilka4 = new Mat();
	                     mLogoMilka4=logotemp.clone();
	                     fillDBORB(mLogoMilka4,vectorMilka4,descriptorMilka4);
	                     hp=HeadPoseStatus.TRACKING;
	                     Log.i("ProcessFrane","After Taking Snapshot ");
	                 }
	        		
	        	   if(hp==HeadPoseStatus.TRACKING)
	                {
	        	   	matcherHamming = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMINGLUT);
	                matcherHamming.add(orbDescriptors);
	                matcherHamming.train();// proba
	              
	                capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
	                Imgproc.resize(mGray, mGray,new Size(480,320)); 
	                orbDetector.detect( mGray, vectorFrame );
	                orbDescriptor.compute(mGray, vectorFrame, descriptorFrame);

	                matcherHamming.match(descriptorFrame, matches); 
	               // Vector<DMatch> matchesXXX = new Vector<DMatch>();
	                MatOfDMatch matchesXXX = new MatOfDMatch();
	                //for (DMatch t : matches)
	                  ///  if(t.distance<BOUNDARY)
	                     //   matchesXXX.add(t);
	                for(int i=0;i < descriptorFrame.rows();i++)
	                {
	                	//if(matches[i].)
	                	if(matches.toList().get(i).distance <BOUNDARY)
	                		matchesXXX.toList().add(matches.toList().get(i));
	                	
	                	
	                }
	                
	                Mat nGray = new Mat();
	                Mat nLogo = new Mat();
	                Mat nRgba = new Mat();
	                Imgproc.cvtColor(mGray, nGray, Imgproc.COLOR_RGBA2RGB, 3);
	                Imgproc.cvtColor(mLogoMilka4, nLogo, Imgproc.COLOR_RGBA2BGR, 3);
	                Features2d.drawMatches(nGray, vectorFrame, nLogo, vectorMilka4, matchesXXX, nRgba);
	                Imgproc.cvtColor(nRgba, mRgba, Imgproc.COLOR_RGB2RGBA, 4);
	                }//TRACKING
	           }	catch(Exception e){
	                    Log.e( "SVK APPLICATION","in ORB "+ e.toString());
	                }
	            break;  
	        

	        }
		 Bitmap bmp;
		 if(!mRgba.empty())
		 {
		 bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);

	        try 
	        {
	        	Utils.matToBitmap(mRgba, bmp);
	        } catch(Exception e) {
	        	Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
	            bmp.recycle();
	            bmp = null;
	        }
	        
	        return bmp;
		 }
		 else return null;
    }

    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mRgba != null)
                mRgba.release();
            if (mGray != null)
                mGray.release();
            if (mCascadeFile != null)
            	mCascadeFile.delete();
            if (mNativeDetector != null)
            	mNativeDetector.release();

            mRgba = null;
            mGray = null;
            mCascadeFile = null;
        }
    }
}
