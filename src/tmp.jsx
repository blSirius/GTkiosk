import React, { useState, useEffect, useRef } from 'react';
import * as faceapi from 'face-api.js';
import axios from 'axios';
import FaceDetectionCSS from './style/FaceDetection.module.css';
import Greeting from './Greeting';

function FaceDetection() {
  const videoHeight = 450;
  const videoWidth = 600;
  const videoRef = useRef();

  let ramState = 0;
  const [ram, setRam] = useState([]);

  //Effect
  //Call Model function and Play Video
  useEffect(() => {
    loadModels();
    videoRef.current.addEventListener('play', runFaceRecognition);
  }, []);
  //Reset Ram value
  useEffect(() => { setInterval(() => { setRam([]) }, 120000); }, []);

  //Download Model
  const loadModels = async () => {
    try {
      await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
        faceapi.nets.faceExpressionNet.loadFromUri('./models'),
        faceapi.nets.ageGenderNet.loadFromUri('./models'),
      ]);
      const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
      videoRef.current.srcObject = stream;
    } catch (error) {
      console.error(error);
    }
  };

  //Get Image and Label
  async function getLabeledFaceDescriptions() {
    try {
      const response = await axios.get('http://localhost:3000/getLabelFolder');
      const { folders } = response.data;

      const labeledFaceDescriptors = await Promise.all(
        folders.map(async (label) => {
          const descriptions = [];
          for (let i = 1; i < 2; i++) {
            const img = await faceapi.fetchImage(
              `http://localhost:3000/getImageFolder/${label}/${i}.png`
            );

            const detections = await faceapi
              .detectSingleFace(img)
              .withFaceLandmarks()
              .withFaceDescriptor();

            if (detections) {
              descriptions.push(detections.descriptor);
            }
          }

          //All Label consist of Bike and Peng, Get descriptor
          return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
      );

      return labeledFaceDescriptors;
    } catch (error) {
      console.error('Error:', error);
      throw new Error('Error fetching labeled face descriptions');
    }
  }

  //Prediction
  const runFaceRecognition = async () => {
    const labeledFaceDescriptors = await getLabeledFaceDescriptions();
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

    const canvas = faceapi.createCanvasFromMedia(videoRef.current);
    document.body.append(canvas);

    const displaySize = { width: videoRef.current.width, height: videoRef.current.height };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
      const detections = await faceapi
        .detectSingleFace(videoRef.current)
        .withFaceLandmarks()
        .withFaceDescriptor()
        .withFaceExpressions()
        .withAgeAndGender();

      if (detections) {
        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
        const results = faceMatcher.findBestMatch(resizedDetections.descriptor);
        const box = resizedDetections.detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: results.label
        });
        drawBox.draw(canvas);

        addFaceData(results, detections, resizedDetections);
      }
      else {
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
      }
    }, 1000);
  };

  //add information and call postDetectedSingleFaceFolder
  const addFaceData = async (results, detections, resizedDetections) => {
    const name = results.label;
    const expression = Object.keys(detections.expressions).reduce((a, b) => detections.expressions[a] > detections.expressions[b] ? a : b);
    const age = Math.round(detections.age);
    const gender = detections.gender;
    const date = new Date().getDate() + "/" + (new Date().getMonth() + 1) + "/" + new Date().getFullYear();
    const time = new Date().toLocaleTimeString();

    const faceImage = await faceapi.extractFaces(videoRef.current, [resizedDetections.detection]);
    const dataURL = faceImage[0].toDataURL();
    const blob = dataURLToBlob(dataURL);
    const file = new File([blob], 'img.jpg', { type: 'image/jpeg' });

    setRam((prev) => {
      const updatedRam = prev.includes(results.label) ? prev : [...prev, results.label];
      if (updatedRam !== prev) {
        ramState = 1;
      }
      return updatedRam;
    });

    if (ramState === 1) {
      postDetectedSingleFaceFolder(file, name, expression, age, gender, date, time);
      ramState = 0; // Reset ramState after processing
    }
  };

  const postFaceDetected = async (name, expression, age, gender, single_img, date, time) => {
    try {
      const res = await axios.post('http://localhost:3000/postFaceDetected', {
        name, expression, age, gender, single_img, date, time
      });
      console.log(res.data);
    } catch (err) {
      console.log(err);
    }
  };

  const postDetectedSingleFaceFolder = async (file, name, expression, age, gender, date, time) => {
    const formData = new FormData();
    formData.append('detectedSingleFace', file);
    formData.append('folderName', name);

    try {
      const res = await axios.post('http://localhost:3000/postDetectedSingleFaceFolder', formData);
      const single_img = res.data;
      if (res) { console.log('postDetectedSingleFaceFolder successfully'); }
      postFaceDetected(name, expression, age, gender, single_img, date, time);
    } catch (error) {
      console.error('Error at postDetectedSingleFaceFolder');
    }
  };

  function dataURLToBlob(dataURL) {
    var arr = dataURL.split(','), mime = arr[0].match(/:(.*?);/)[1],
      bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], { type: mime });
  }

  return (
    <div>
      {/* video */}
      <div className={FaceDetectionCSS.frame}>
        <video className={FaceDetectionCSS.video} ref={videoRef} autoPlay
          muted height={videoHeight} width={videoWidth}
        ></video>
      </div>

      <div>
        <Greeting ram={ram} />
      </div>

    </div>
  );
}
export default FaceDetection;