#include <mitsuba/core/properties.h> // Technically not needed if no specific cpp logic uses it
#include <mitsuba/core/warp.h>     // Technically not needed if no specific cpp logic uses it
#include <mitsuba/render/bsdf.h>   // Good to include for context, but Base types pulled by .h
// Adjust include path as necessary
#include "NeuralBRDF.h" // The header file created above

MTS_NAMESPACE_BEGIN

MTS_IMPLEMENT_CLASS_VARIANT(NeuralBRDF, BSDF, "neural_brdf")
MTS_EXPORT_PLUGIN(NeuralBRDF, "Neural BRDF Plugin")

MTS_NAMESPACE_END