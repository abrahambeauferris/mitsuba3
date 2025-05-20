#pragma once

#include <mitsuba/render/bsdf.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h> // For string::split
#include <mitsuba/core/warp.h>   // For warp::square_to_cosine_hemisphere

// It's crucial to have npy.hpp accessible.
// This typically comes from the cnpy library.
// The Neural-BRDF repository includes a version of it.
// Ensure it's correctly placed, e.g., in this directory or a common include path.
#include "npy.hpp"

// --- Start of Minimal djb::vec3 and geometric utilities ---
// These are minimal, static helper functions inspired by dj_brdf_mod.h
// to perform the necessary geometric transformations for feature calculation,
// without needing the entire dj_brdf_mod.h library.
namespace djb {
    // A simple 3D vector class compatible with Dr.Jit types if needed, but float for now.
    // In a full Dr.Jit integration, these would use dr::Array<Float, 3> etc.
    // For clarity and directness from nbrdf_npy.cpp, using basic float here.
    struct vec3 {
        float x, y, z;
        vec3(float _x = 0.f, float _y = 0.f, float _z = 0.f) : x(_x), y(_y), z(_z) {}

        // Constructor from Mitsuba/Dr.Jit Vector3f
        template <typename Vector3f_>
        vec3(const Vector3f_& v) : x(dr::x(v)), y(dr::y(v)), z(dr::z(v)) {}

        // Conversion to Mitsuba/Dr.Jit Vector3f
        template <typename Vector3f_>
        Vector3f_to_mitsuba() const { return Vector3f_(x, y, z); }
    };

    static inline float dot(const vec3& a, const vec3& b) {
        return (a.x * b.x + a.y * b.y + a.z * b.z);
    }

    static inline vec3 normalize(const vec3& v) {
        float mag_sqr = dot(v, v);
        if (mag_sqr <= 1e-12f)
            return vec3(0.f, 0.f, 0.f);
        float inv_mag = 1.f / std::sqrt(mag_sqr);
        return vec3(v.x * inv_mag, v.y * inv_mag, v.z * inv_mag);
    }

    static inline vec3 cross(const vec3& a, const vec3& b) {
        return vec3(a.y * b.z - a.z * b.y,
                    a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x);
    }
    
    // Simplified xyz_to_theta_phi using std::acos, std::atan2 for broad compatibility
    static inline void xyz_to_theta_phi(const vec3& p, float &theta, float &phi) {
        if (p.z > 0.99999f) {
            theta = 0.0f;
            phi = 0.0f;
        } else if (p.z < -0.99999f) {
            theta = (float)M_PI; // Use M_PI from cmath
            phi = 0.0f;
        } else {
            theta = std::acos(p.z);
            phi = std::atan2(p.y, p.x);
        }
    }

    static inline vec3 rotate_vector(const vec3& v_in, const vec3& axis, float angle) {
        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);
        float dot_axis_v = dot(axis, v_in);
        
        vec3 term1 = vec3(v_in.x * cos_angle, v_in.y * cos_angle, v_in.z * cos_angle);
        vec3 term2 = vec3(axis.x * dot_axis_v * (1.f - cos_angle),
                          axis.y * dot_axis_v * (1.f - cos_angle),
                          axis.z * dot_axis_v * (1.f - cos_angle));
        vec3 cross_av = cross(axis, v_in);
        vec3 term3 = vec3(cross_av.x * sin_angle, cross_av.y * sin_angle, cross_av.z * sin_angle);

        return vec3(term1.x + term2.x + term3.x,
                    term1.y + term2.y + term3.y,
                    term1.z + term2.z + term3.z);
    }

    // Static version of io_to_hd from djb::brdf
    static inline void io_to_hd(const vec3& i_light, const vec3& o_viewer, vec3 *h_out, vec3 *d_out) {
        const vec3 y_axis_djb = vec3(0.f, 1.f, 0.f);
        const vec3 z_axis_djb = vec3(0.f, 0.f, 1.f);
        float theta_h_val, phi_h_val;

        *h_out = normalize(vec3(i_light.x + o_viewer.x, i_light.y + o_viewer.y, i_light.z + o_viewer.z));
        xyz_to_theta_phi(*h_out, theta_h_val, phi_h_val);
        
        // In asztr/Neural-BRDF's mitsuba/nbrdf_npy.cpp, for isotropic materials,
        // phi_h is effectively set to 0 for feature calculation by using hx = sin(theta_h), hy = 0, hz = cos(theta_h).
        // The 'd' vector is then i_light rotated by -theta_h around Y.
        // This simplified 'd' is what's used if phi_h is zeroed out *before* rotation.

        // Original djb logic for 'd':
        vec3 tmp = rotate_vector(i_light, z_axis_djb, -phi_h_val);
        *d_out = normalize(rotate_vector(tmp, y_axis_djb, -theta_h_val));
    }
} // namespace djb
// --- End of Minimal djb::vec3 and geometric utilities ---


MTS_NAMESPACE_BEGIN

template <typename Float, typename Spectrum>
class NeuralBRDF final : public BSDF<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(BSDF, component_count, flags, m_flags)
    MTS_IMPORT_TYPES(Texture)

    NeuralBRDF(const Properties &props) : Base(props) {
        std::string npy_path_prefix = props.string("npy_path");
        if (npy_path_prefix.empty())
            Throw("The 'npy_path' (prefix for .npy weight files, e.g. 'path/to/material_') must be specified!");

        // Files are expected to be named like: material_w1.npy, material_b1.npy, etc.
        std::string w1_path = npy_path_prefix + "w1.npy";
        std::string b1_path = npy_path_prefix + "b1.npy";
        std::string w2_path = npy_path_prefix + "w2.npy";
        std::string b2_path = npy_path_prefix + "b2.npy";
        std::string w3_path = npy_path_prefix + "w3.npy";
        std::string b3_path = npy_path_prefix + "b3.npy";
        
        std::vector<unsigned long> shape_w1, shape_b1, shape_w2, shape_b2, shape_w3, shape_b3;
        
        try {
            npy::LoadArrayFromNumpy(w1_path, shape_w1, m_w1_data);
            npy::LoadArrayFromNumpy(b1_path, shape_b1, m_b1_data);
            npy::LoadArrayFromNumpy(w2_path, shape_w2, m_w2_data);
            npy::LoadArrayFromNumpy(b2_path, shape_b2, m_b2_data);
            npy::LoadArrayFromNumpy(w3_path, shape_w3, m_w3_data);
            npy::LoadArrayFromNumpy(b3_path, shape_b3, m_b3_data);
        } catch (const std::runtime_error& e) {
            Throw("Failed to load NBRDF weights. Ensure 'npy_path' is correct prefix and files like '%sw1.npy' exist. Original error: %s", npy_path_prefix.c_str(), e.what());
        }

        // Verify shapes (optional but good practice)
        // Expected: w1: (6,21), b1: (21), w2: (21,21), b2: (21), w3: (21,3), b3: (3)
        // Note: npy::LoadArrayFromNumpy loads flattened data. The .npy files from h5_to_npy.py are already transposed for TF convention.
        // We need to ensure our access matches this. The `nbrdf_npy.cpp` example accesses W1[i*21+j] (row-major access if W1 is 6x21) or W1(i,j) for Eigen.
        // The `h5_to_npy.py` transposes weights: `curr_weight = weights.detach().cpu().numpy().T`
        // Keras dense layers weights are (input_dim, output_dim). Transposed: (output_dim, input_dim).
        // So w1.npy (fc1.weight.T) is (21, 6). b1.npy (fc1.bias) is (21).
        // w2.npy (fc2.weight.T) is (21, 21). b2.npy (fc2.bias) is (21).
        // w3.npy (fc3.weight.T) is (3, 21). b3.npy (fc3.bias) is (3).

        // Let's store expected dimensions for clarity during forward pass
        m_dim_w1 = {21, 6}; m_dim_b1 = {21};
        m_dim_w2 = {21, 21}; m_dim_b2 = {21};
        m_dim_w3 = {3, 21}; m_dim_b3 = {3};

        // Validate loaded sizes
        if (m_w1_data.size() != (m_dim_w1[0] * m_dim_w1[1])) Throw("w1.npy has unexpected size.");
        if (m_b1_data.size() != m_dim_b1[0]) Throw("b1.npy has unexpected size.");
        // ... similar checks for w2, b2, w3, b3 ...

        m_roughness = props.float_("roughness", 0.15f);
        // IMPORTANT: The provided `nbrdf_npy.cpp` and its `Net::forward` do not explicitly use
        // a roughness parameter passed at runtime directly into the 6-feature vector for the MLP.
        // This implies either:
        // 1. The loaded .npy weights are *already* for a specific roughness.
        // 2. Roughness is implicitly handled by one of the 6 geometric input features (e.g., a feature definition is roughness-dependent).
        // 3. The NBRDF model from asztr/Neural-BRDF might not vary with a 'roughness' parameter in this specific plugin version,
        //    and different materials (different .npy sets) represent different appearances including roughness.
        // The guide [source 133] lists "Roughness a" as a potential 5th input feature.
        // For this MVP, we load `m_roughness` but the feature calculation below, directly
        // adapted from `nbrdf_npy.cpp`, does NOT explicitly insert it as one of the 6 inputs to the MLP.
        // If your specific NBRDF model expects roughness as one of the 6 direct inputs,
        // you MUST modify the `features_std` vector in `eval()` accordingly.

        m_scale = props.texture<Spectrum>("scale", Spectrum(1.0f));

        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide;
        if (m_scale->is_spatially_varying()) {
            m_flags |= BSDFFlags::SpatiallyVarying;
        }
        dr::set_attr(this, "m_flags", m_flags);
        
        m_components.push_back(m_flags);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float /* sample1 */, // sample1 is often for component selection, not used here
                                             const Point2f &sample2,
                                             Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        BSDFSample3f bs = dr::zeros<BSDFSample3f>();

        active &= cos_theta_i > 0.f;
        if (dr::none_or<false>(active) || !ctx.is_enabled(BSDFFlags::GlossyReflection))
            return { bs, 0.f };

        bs.wo = warp::square_to_cosine_hemisphere(sample2);
        bs.pdf = warp::square_to_cosine_hemisphere_pdf(bs.wo);
        bs.eta = 1.f;
        bs.sampled_type = +BSDFFlags::GlossyReflection;
        bs.sampled_component = 0;
        
        active &= Frame3f::cos_theta(bs.wo) > 0.f; // Ensure sampled wo is in the upper hemisphere
        if (dr::none_or<false>(active))
             return { bs, 0.f };

        Spectrum value = eval(ctx, si, bs.wo, active);
        
        // PDF cannot be zero if active
        Spectrum result_val = value / dr::select(active && bs.pdf > dr::Epsilon<Float>, bs.pdf, dr::Infinity<Float>);
        
        return { bs, dr::select(active, result_val, 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi); // Mitsuba's wi = outgoing / view
        Float cos_theta_o = Frame3f::cos_theta(wo);    // Mitsuba's wo = incoming / light

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;
        if (dr::none_or<false>(active) || !ctx.is_enabled(BSDFFlags::GlossyReflection))
            return 0.f;

        // Map Mitsuba vectors to djb convention for feature calculation
        // djb's 'o' (outgoing/viewer) is Mitsuba's si.wi
        // djb's 'i' (incoming/light) is Mitsuba's wo
        djb::vec3 djb_o_viewer(si.wi);
        djb::vec3 djb_i_light(wo);
        djb::vec3 h_djb, d_djb; // These will be djb::vec3

        // This computes h_djb and d_djb based on light and view vectors
        djb::io_to_hd(djb_i_light, djb_o_viewer, &h_djb, &d_djb);

        float theta_h_float, phi_h_float;
        djb::xyz_to_theta_phi(h_djb, theta_h_float, phi_h_float);
        phi_h_float = 0.f; // Isotropic NBRDF: rotate frame so phi_h = 0 for feature input

        // These are the 6 input features for the neural network,
        // matching the structure from the original nbrdf_npy.cpp
        std::vector<Float> features_std(6);
        features_std[0] = std::sin(theta_h_float); // Simplified h.x (since phi_h=0)
        features_std[1] = 0.f;                     // Simplified h.y (since phi_h=0)
        features_std[2] = std::cos(theta_h_float); // Simplified h.z
        features_std[3] = d_djb.x;
        features_std[4] = d_djb.y;
        features_std[5] = d_djb.z;
        // Note: Roughness m_roughness is not explicitly passed here. This feature set is
        // directly from nbrdf_npy.cpp. If your NBRDF *requires* roughness as a direct input,
        // one of these features (or an additional one if model differs) must be m_roughness
        // or derived from it. The current assumption is the .npy files are for a fixed roughness
        // or roughness is implicitly part of the other geometric features as trained.

        // --- Manual MLP Forward Pass (using Dr.Jit types where appropriate for active mask) ---
        // Layer 1: (Input: 6, Output: 21), ReLU
        std::vector<Float> h1_vals(m_dim_b1[0]); // 21
        for (size_t j = 0; j < m_dim_b1[0]; ++j) { // Output neuron index (0 to 20)
            Float sum = m_b1_data[j];
            for (size_t k = 0; k < m_dim_w1[1]; ++k) { // Input feature index (0 to 5)
                // m_w1_data is (output_dim, input_dim) = (21, 6)
                sum += features_std[k] * m_w1_data[j * m_dim_w1[1] + k];
            }
            h1_vals[j] = dr::maximum(0.f, sum); // ReLU
        }

        // Layer 2: (Input: 21, Output: 21), ReLU
        std::vector<Float> h2_vals(m_dim_b2[0]); // 21
        for (size_t j = 0; j < m_dim_b2[0]; ++j) { // Output neuron index
            Float sum = m_b2_data[j];
            for (size_t k = 0; k < m_dim_w2[1]; ++k) { // Input neuron index from h1
                // m_w2_data is (output_dim, input_dim) = (21, 21)
                sum += h1_vals[k] * m_w2_data[j * m_dim_w2[1] + k];
            }
            h2_vals[j] = dr::maximum(0.f, sum); // ReLU
        }

        // Layer 3 (Output): (Input: 21, Output: 3), Activation: exp(x) - 1, then max(0,x)
        Float rgb_out[3];
        for (size_t j = 0; j < m_dim_b3[0]; ++j) { // Output RGB channel index (0 to 2)
            Float sum = m_b3_data[j];
            for (size_t k = 0; k < m_dim_w3[1]; ++k) { // Input neuron index from h2
                // m_w3_data is (output_dim, input_dim) = (3, 21)
                sum += h2_vals[k] * m_w3_data[j * m_dim_w3[1] + k];
            }
            // Activation from nbrdf_npy.cpp's Net::forward
            rgb_out[j] = dr::maximum(0.f, dr::exp(sum) - 1.f);
        }
        
        UnpolarizedSpectrum result_spec = dr::load<Spectrum>(rgb_out, active);

        return dr::select(active, result_spec * m_scale->eval(si, active) * cos_theta_o, 0.f);
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        Float cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;
        if (dr::none_or<false>(active) || !ctx.is_enabled(BSDFFlags::GlossyReflection))
            return 0.f;

        // PDF of cosine-weighted hemispherical sampling
        return dr::select(active, warp::square_to_cosine_hemisphere_pdf(wo), 0.f);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "NeuralBRDF[" << std::endl
            // Intentionally not printing all weights to keep string short
            << "  npy_path_prefix = " << "\"" << "loaded_from_constructor" << "\"," << std::endl
            << "  roughness = " << m_roughness << "," << std::endl
            << "  scale = " << string::indent(m_scale->to_string()) << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    std::vector<float> m_w1_data, m_b1_data;
    std::vector<float> m_w2_data, m_b2_data;
    std::vector<float> m_w3_data, m_b3_data;

    // Expected dimensions after loading (transposed weights from Keras)
    std::vector<size_t> m_dim_w1, m_dim_b1;
    std::vector<size_t> m_dim_w2, m_dim_b2;
    std::vector<size_t> m_dim_w3, m_dim_b3;
    
    Float m_roughness;
    ref<Texture> m_scale;
};

MTS_INTERNAL_DECLARE_CLASS_VARIANT(NeuralBRDF, BSDF)
MTS_NAMESPACE_END