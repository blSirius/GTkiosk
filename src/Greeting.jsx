import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col } from 'react-bootstrap';

import GreetingCSS from './style/Greeting.module.css';

function Greeting({ram}) {
    const [faceDetections, setFaceDetections] = useState([]);

    useEffect(() => {
        fetchFaceDetections();
    },[ram]);

    const fetchFaceDetections = async () => {
        console.log(1);
        try {
            const response = await axios.get('http://localhost:3000/getFaceDetected');
            setFaceDetections(response.data);
        } catch (error) {
            console.error('Error fetching face detections:', error);
        }
    };

    const getImagePath = (name, single_img) => {
        const trimmedName = name.trim();
        return `http://localhost:3000/getDetectedSingleFaceFolder/${encodeURIComponent(trimmedName)}/${single_img}`;
    };

    return (
        <Container>
            {ram}
            <h2 className={GreetingCSS.test} style={{ textAlign: 'center' }}>Welcome Home Henry</h2>


            {faceDetections.map((data, index) => (

                // justify-content-md-center m-2 p-2

                <Row className={GreetingCSS.box} key={index}>

                    <p className={GreetingCSS.posi_time_right} style={{ paddingRight: '48px' }}><b>Time:</b>{data.time}</p>

                    <hr className={GreetingCSS.stylehr} />
                    {/* <hr className={GreetingCSS.stylehr}> */}

                    <Row style={{ margin: '20px', fontSize: '22px', justifyContent: 'center' }}>
                        <Col >
                            <img width={100} height={100} style={{ borderRadius: '0.5rem' }} src={getImagePath(data.name, data.single_img)} alt="" />
                        </Col>
                        <Col className={GreetingCSS.flex}>
                            <Row className={GreetingCSS.flex}>
                                <p><b>Name:</b> {data.name}</p>
                                <p><b>Expression:</b> {data.expression}</p>
                            </Row>
                        </Col>
                        <Col className={GreetingCSS.flex}>
                            <Row className={GreetingCSS.flex}>
                                <p><b>Age:</b> {data.age}</p>
                                <p><b>Gender:</b> {data.gender}</p>
                            </Row>

                        </Col>
                        <Col className={GreetingCSS.font_greeting} >
                            <b>{data.greeting}</b>
                        </Col>
                    </Row>
                </Row>
            ))}
        </Container>
    );
}
export default Greeting;