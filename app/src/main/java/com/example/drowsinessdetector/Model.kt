package com.example.drowsinessdetector

data class PredictionResult(
    val label: String,
    val confidence: Float,
    val features: List<Float>
)