//ChatGPT was utilized to refine, format, setup of flask server, and debug this code
import DragDropUploader from "../components/DragDropUploader"
import { Box, Button, Snackbar, Alert, LinearProgress, Card, CardContent, Typography, CircularProgress } from "@mui/material";
import { createTheme } from "@mui/material/styles";
import CheckIcon from '@mui/icons-material/Check';
import CancelIcon from '@mui/icons-material/Cancel';
import { useState } from "react";
import axios from "axios";

const theme = createTheme({
    palette: {
        mode: "dark",
    },
});

export default function Predict() {
    const [image, setImage] = useState(null);
    const [error, setError] = useState("");
    const [openSnackbar, setOpenSnackbar] = useState(false);
    const [loading, setLoading] = useState(false);
    const [prediction, setPrediction] = useState(null);
    const [confidence, setConfidence] = useState(null);


    const handleSubmit = async () => {
        if (!image) {
            setError("Please upload an image first!");
            setOpenSnackbar(true);
            return;
        }

        setLoading(true);

        const formData = {
            file: image.file
        }

        try {
            const response = await axios.post("http://localhost:5001/predict", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            const result = response.data;
            console.log(result);

            if (result.prediction == "yes" || result.prediction == "no") {
                console.log(result);
                setPrediction(result.prediction);
                setConfidence(result.confidence);
                setOpenSnackbar(true);
            } else {
                setError(result.message || "something went wrong");
                setOpenSnackbar(true);
            }
        } catch (err) {
            setError("error submitting the image");
            console.error(err);
            setOpenSnackbar(true);
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <div className="container mt-5" style={{ height: "100%" }}>
                <div className="text-center my-5">
                    <h2>Brain Tumour Detection</h2>

                </div>
                <div className="row">
                    <div className="col-lg-6">
                        <div className="mt-5 mb-3">
                            <DragDropUploader setImage={setImage} image={image} setOpenSnackbar={setOpenSnackbar} setError={setError} error={error} />
                        </div>
                        <Button
                            variant="contained"
                            color="primary"
                            onClick={handleSubmit}
                            sx={{ marginBottom: 2, backgroundColor: "primary.main", color: "text.primary", '&:hover': { backgroundColor: "primary.dark", }, }}
                        >
                            Analyze MRI
                        </Button>
                    </div>
                    <div className="col-lg-6 d-flex justify-content-lg-end">
                        <Card sx={{ width: "100%", maxWidth: 500, marginTop: 2 }} style={{ backgroundColor: '#262b2e', color: 'white', borderRadius: '15px' }}>
                            <CardContent style={{ backgroundColor: '#262b2e', display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}>
                                <Typography variant="h6" gutterBottom>
                                    MRI analysis
                                </Typography>
                                {image?.imageUrl && (
                                    <Box mt={2}>
                                        <img
                                            src={image.imageUrl}
                                            alt="Preview"
                                            style={{ maxWidth: "100%", borderRadius: "8px" }}
                                        />
                                    </Box>
                                )}

                                {loading && (
                                    <Box mt={2}>
                                        <LinearProgress />
                                    </Box>
                                )}

                                {prediction !== null && (
                                    <Box mt={2}>
                                        <Typography variant="h6">
                                            {prediction === "yes" ? (
                                                <>
                                                    <CheckIcon sx={{ color: "green", marginRight: 1 }} />
                                                    Tumor is present
                                                </>
                                            ) : (
                                                <>
                                                    <CancelIcon sx={{ color: "red", marginRight: 1 }} />
                                                    No tumor detected
                                                </>
                                            )}
                                        </Typography>
                                    </Box>
                                )}

                                {confidence !== null && (
                                    <Box mt={2}>
                                        <Typography variant="subtitle1" gutterBottom>
                                            Confidence Level: {(confidence * 100).toFixed(2)}%
                                        </Typography>
                                        <CircularProgress
                                            variant="determinate"
                                            value={confidence * 100}  
                                            size={80}
                                            thickness={5}
                                            sx={{ color: confidence >= 0.5 ? "green" : "red" }}  
                                        />
                                    </Box>
                                )}
                            </CardContent>
                        </Card>

                    </div>
                </div>

            </div>

            <Snackbar
                open={openSnackbar}
                autoHideDuration={3000}
                onClose={() => setOpenSnackbar(false)}
            >
                <Alert severity={error ? "error" : "success"} onClose={() => setOpenSnackbar(false)}>
                    {error || "Image uploaded successfully!"}
                </Alert>
            </Snackbar>
        </>
    )
}
