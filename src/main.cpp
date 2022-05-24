/*
 * Forward-mode autodiff test with Sphere function
 *
 * $CXX -Wall -std=c++17 -mcpu=native -O3 -ffp-contract=fast -I$EIGEN_INCLUDE_PATH -I$AUTODIFF_INCLUDE_PATH -I$OPTIM/include autodiff_forward_sphere.cpp -o autodiff_forward_sphere.test -L$OPTIM -loptim -framework Accelerate
 */
#include <iostream>
#include <optional>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

//#include <Eigen/Core>

#include <math.h>

using namespace autodiff;
using namespace Eigen;
namespace py = pybind11;

static const double gravity_acceleration = 9.81;
static const double v_max_ms = 100.0;
static const double v_epsilon_ms = 0.01;
static const double watts_epsilon_w = 0.1;

inline static double air_density(double altitude_m, double temperature_C)
{
    double pressure_pa = 100 * 1013.25 * std::pow(1 - 0.0065 * altitude_m / 288.15, 5.255);
    double R_air = 287;
    double temperature_K = temperature_C + 273;
    return pressure_pa / temperature_K / R_air;
}

struct CPModelParams
{
    double anaerobic_capacity_J;
    double threshold_power_W;
    CPModelParams(double anaerobic_capacity_J, double threshold_power_W) : anaerobic_capacity_J(anaerobic_capacity_J),
                                                                           threshold_power_W(threshold_power_W) {}
};
struct PhysicsParams
{
    double total_mass_kg;
    double drivetrain_efficiency;
    double air_penetration_coefficient_m2;
    double rolling_resistance_coefficient;
    PhysicsParams(double total_mass_kg,
                  double drivetrain_efficiency,
                  double air_penetration_coefficient_m2,
                  double rolling_resistance_coefficient) : total_mass_kg(total_mass_kg),
                                                           drivetrain_efficiency(drivetrain_efficiency),
                                                           air_penetration_coefficient_m2(air_penetration_coefficient_m2),
                                                           rolling_resistance_coefficient(rolling_resistance_coefficient)
    {
    }
};
struct SegmentParams
{
    int size;
    Eigen::VectorXd segment_length_m;
    Eigen::VectorXd grade_percent;
    Eigen::VectorXd altitude_m;
    Eigen::VectorXd temperature_C;
    Eigen::VectorXd wind_speed_ms;

    PhysicsParams physics_params;

    // Computed properties
    double rolling_resistance_force_N;
    Eigen::VectorXd air_density_kgm3;
    Eigen::VectorXd gravity_force_N;
    Eigen::VectorXd air_resistance_coef;

    void compute_properties()
    {
        rolling_resistance_force_N = physics_params.rolling_resistance_coefficient * physics_params.total_mass_kg * gravity_acceleration;
        gravity_force_N = Eigen::VectorXd(size);
        air_resistance_coef = Eigen::VectorXd(size);
        air_density_kgm3 = Eigen::VectorXd(size);
        for (int i = 0; i < size; ++i)
        {
            air_density_kgm3[i] = air_density(altitude_m[i], temperature_C[i]);
            gravity_force_N[i] = physics_params.total_mass_kg * gravity_acceleration * sin(atan(grade_percent[i] / 100));
            air_resistance_coef[i] = 0.5 * air_density_kgm3[i] * physics_params.air_penetration_coefficient_m2;
        }
    }
    SegmentParams(PhysicsParams physics_params,
                  Eigen::VectorXd segment_length_m,
                  Eigen::VectorXd grade_percent,
                  Eigen::VectorXd altitude_m,
                  Eigen::VectorXd temperature_C,
                  Eigen::VectorXd wind_speed_ms) : physics_params(physics_params),
                                                   size(segment_length_m.size()),
                                                   segment_length_m(segment_length_m),
                                                   grade_percent(grade_percent),
                                                   altitude_m(altitude_m),
                                                   temperature_C(temperature_C),
                                                   wind_speed_ms(wind_speed_ms)
    {
        if (grade_percent.size() != size || altitude_m.size() != size || temperature_C.size() != size || wind_speed_ms.size() != size)
        {
            throw std::runtime_error("SegmentParams: wrong array size.");
        }
        compute_properties();
    }
};

SegmentParams create_segment_params(const PhysicsParams &physics_params,
                                    double _segment_length_m,
                                    double _grade_percent,
                                    double _altitude_m,
                                    double _temperature_C,
                                    double _wind_speed_ms)
{
    Eigen::VectorXd segment_length_m(1);
    Eigen::VectorXd grade_percent(1);
    Eigen::VectorXd altitude_m(1);
    Eigen::VectorXd temperature_C(1);
    Eigen::VectorXd wind_speed_ms(1);
    return SegmentParams(physics_params, segment_length_m, grade_percent, altitude_m, temperature_C, wind_speed_ms);
}

typedef var (*autodiff_func)(const ArrayXvar &, const SegmentParams &, const CPModelParams &);

template <typename T>
static inline T power_from_speed(T speed, int i, const SegmentParams &segment_params)
{
    double wind_speed = segment_params.wind_speed_ms[i];
    double rolling_resistance_force = segment_params.rolling_resistance_force_N;
    double gravity_force = segment_params.gravity_force_N[i];
    double air_resistance_coef = segment_params.air_resistance_coef[i];
    T air_resistance_force = air_resistance_coef * abs(speed + wind_speed) * (speed + wind_speed);
    return (1 / segment_params.physics_params.drivetrain_efficiency * speed * (air_resistance_force + rolling_resistance_force + gravity_force));
}

template <typename arrayT>
arrayT power_from_speed(const arrayT &speed, const SegmentParams &segment_params)
{
    int n = speed.size();
    assert(speed.size() == segment_params.size);
    arrayT out(n);
    for (int i = 0; i < n; ++i)
    {
        auto prev_speed = (i == 0) ? 0 : speed[i - 1];
        auto duration = segment_params.segment_length_m[i] / speed[i];
        auto power = power_from_speed(speed[i], i, segment_params);
        out[i] = power;
    }
    return out;
}

ArrayXd py_power_from_speed(const ArrayXd &speed, const SegmentParams &segment_params)
{
    return power_from_speed(speed, segment_params);
}

static inline double speed_from_power(double power, int i, const SegmentParams &segment_params, double prev_speed = 0)
{
    double a = 0, b = v_max_ms;
    while (b - a > v_epsilon_ms)
    {
        double c = (b + a) / 2;
        double est_power = power_from_speed(c, i, segment_params);
        if (est_power > power)
        {
            b = c;
        }
        else
        {
            a = c;
        }
    }
    return b;
}

VectorXd py_speed_from_power(const VectorXd &power, const SegmentParams &segment_params)
{
    int n = power.size();
    VectorXd out(n);
    for (int i = 0; i < n; ++i)
    {
        auto prev_speed = (i == 0) ? 0 : out[i - 1];
        out[i] = speed_from_power(power[i], i, segment_params, prev_speed);
    }
    return out;
}

var new_wp_bal(var last_wp_bal, double wp_0, var delta_p, var duration)
{
    var wp_bal = last_wp_bal + condition(delta_p >= 0, -duration * delta_p,
                                         (wp_0 - last_wp_bal) * (1 - exp(delta_p * duration / wp_0)));
    return min(wp_0, wp_bal);
}

double new_wp_bal(double last_wp_bal, double wp_0, double delta_p, double duration)
{
    double wp_bal = last_wp_bal;
    if (delta_p >= 0)
    {
        wp_bal -= duration * delta_p;
    }
    else
    {
        wp_bal += (wp_0 - last_wp_bal) * (1 - exp(delta_p * duration / wp_0));
        if (wp_bal > wp_0)
        {
            wp_bal = wp_0;
        }
    }
    return wp_bal;
}

template <typename arrayT>
static inline arrayT wp_bal(const arrayT &target_speed, const SegmentParams &segment_params, const CPModelParams &cp_model_params)
{
    int n = segment_params.size;
    arrayT out(n + 1);
    arrayT power = power_from_speed(target_speed, segment_params);
    double wp_0 = cp_model_params.anaerobic_capacity_J;
    auto last_wp_bal = wp_0;
    auto wp_bal_min = wp_0;
    out[0] = wp_0;
    for (int i = 0; i < n; ++i)
    {
        auto delta_p = power[i] - cp_model_params.threshold_power_W;
        auto duration = segment_params.segment_length_m[i] / target_speed[i];
        out[i + 1] = new_wp_bal(out[i], wp_0, delta_p, duration);
    }
    return out.tail(n);
}

template <typename arrayT>
static inline arrayT average_speed(const arrayT &target_speed, const SegmentParams &segment_params)
{
    int n = segment_params.size;
    arrayT out(n);
    arrayT power = power_from_speed(target_speed, segment_params);
    for (int i = 0; i < n; ++i)
    {
        auto prev_speed = i == 0 ? 0 : target_speed(i - 1);
        auto prev_f = i == 0 ? 0 : power_from_speed(prev_speed, i, segment_params) / target_speed(i);
        auto a = (power(i) / target_speed(i) - prev_f) / segment_params.physics_params.total_mass_kg;
        auto avg_speed = 0.5 * (target_speed(i) + prev_speed);
        for (int j = 0; j < 4; ++j)
        {
            auto duration = segment_params.segment_length_m(i) / avg_speed;
            avg_speed = target_speed(i) - pow(target_speed(i) - prev_speed, 2) / a / duration;
        }
        out(i) = avg_speed;
    }
    return out;
}

VectorXd py_average_speed(const VectorXd &target_speed, const SegmentParams &segment_params, std::optional<py::array_t<double>> jac)
{
    int n = segment_params.size;

    if (jac.has_value())
    {
        VectorXvar speed_var(target_speed);
        VectorXvar average_speed_array = average_speed(speed_var, segment_params);
        auto J = jac.value();

        VectorXd out(n);
        for (int i = 0; i < n; i++)
        {
            out[i] = val(average_speed_array[i]);
            for (int j = 0; j < n; j++)
            {
                J.mutable_at(i, j) = derivatives(average_speed_array(i), wrt(speed_var(j)))[0];
            }
        }
        return out;
    }
    else
    {
        return average_speed(target_speed, segment_params);
    }
}

double py_total_time(const VectorXd &target_speed, const SegmentParams &segment_params)
{
    auto average_speed_array = average_speed(target_speed, segment_params);
    double out = 0;
    int n = segment_params.size;
    for (int i = 0; i < n; ++i)
    {
        out += segment_params.segment_length_m(i) / average_speed_array(i);
    }

    return out;
}

VectorXd py_wp_bal(const VectorXd &target_speed, const SegmentParams &segment_params,
                   const CPModelParams &cp_model_params, std::optional<py::array_t<double>> jac)
{
    int n = segment_params.size;

    if (jac.has_value())
    {
        VectorXvar speed_var(target_speed);
        VectorXvar wp_bal_array = wp_bal(speed_var, segment_params, cp_model_params);
        auto J = jac.value();

        VectorXd out(n);
        for (int i = 0; i < n; i++)
        {
            out[i] = val(wp_bal_array[i]);
            for (int j = 0; j < n; j++)
            {
                J.mutable_at(i, j) = derivatives(wp_bal_array(i), wrt(speed_var(j)))[0];
            }
        }
        return out;
    }
    else
    {
        return wp_bal(target_speed, segment_params, cp_model_params);
    }
}

PYBIND11_MODULE(_core, m)
{

    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("wp_bal", py_wp_bal, py::arg("speed"), py::arg("segment_params"),
          py::arg("cp_model_params"), py::arg("jac") = std::nullopt);
    m.def("average_speed", py_average_speed, py::arg("speed"), py::arg("segment_params"), py::arg("jac") = std::nullopt);
    m.def("total_time", py_total_time, py::arg("speed"), py::arg("segment_params"));
    m.def("speed_from_power", py_speed_from_power);
    m.def("power_from_speed", py_power_from_speed);
    py::class_<CPModelParams>(m, "CPModelParams")
        .def(py::init<double, double>(), py::arg("wp0"), py::arg("cp"));
    py::class_<PhysicsParams>(m, "PhysicsParams")
        .def(py::init<double, double, double, double>(), py::arg("total_mass_kg"),
             py::arg("drivetrain_efficiency"), py::arg("air_penetration_coefficient_m2"), py::arg("rolling_resistance_coefficient"));
    py::class_<SegmentParams>(m, "SegmentParams")
        .def(py::init<PhysicsParams,
                      Eigen::VectorXd,
                      Eigen::VectorXd,
                      Eigen::VectorXd,
                      Eigen::VectorXd,
                      Eigen::VectorXd>());
}
