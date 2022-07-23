use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::{
    buffer::{CpuBufferPool, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo, RenderingInfo,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, QueueCreateInfo,
    },
    image::{view::ImageView, ImageAccess, ImageUsage, SwapchainImage},
    impl_vertex,
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            render_pass::PipelineRenderingCreateInfo,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{LoadOp, StoreOp},
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, Window, WindowBuilder},
};

fn main() {
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: vulkano_win::required_extensions(),
        ..Default::default()
    })
    .expect("Failed to create instance!");

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title("Vulkan")
        .with_fullscreen(Some(Fullscreen::Borderless(None)))
        .build_vk_surface(&event_loop, instance.clone())
        .expect("Failed to build surface!");

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.api_version() >= Version::V1_3)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .expect("No suitable physical device found!");

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: Features {
                dynamic_rendering: true,
                ..Features::none()
            },
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .expect("Failed to create device!");

    let queue = queues.next().expect("Failed to create queue!");

    let (mut swapchain, images) = {
        let surface_capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("Failed to obtain surface capabilities!");

        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .expect("Failed to configure image format!")[0]
                .0,
        );

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: surface.window().inner_size().into(),
                image_usage: ImageUsage::color_attachment(),
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .expect("Failed to obtain supported composite alpha!"),
                ..Default::default()
            },
        )
        .expect("Failed to create swapchain!")
    };

    #[repr(C)]
    #[derive(Clone, Copy, Default, Pod, Zeroable)]
    struct Vertex {
        position: [f32; 2],
        color: [f32; 4],
    }
    impl_vertex!(Vertex, position, color);

    let vertices = [
        Vertex {
            position: [-0.5, -0.25],
            color: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.0, 0.5],
            color: [0.0, 1.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.25, -0.1],
            color: [0.0, 0.0, 1.0, 1.0],
        },
    ];

    let buffer_pool = CpuBufferPool::vertex_buffer(device.clone());
    let vertex_buffer = buffer_pool
        .chunk(vertices)
        .expect("Failed to create vertex buffer!");

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
                #version 460
                layout(location = 0) in vec2 position;
                layout(location = 1) in vec4 color;
                layout(location = 0) out vec4 vertColor;
                
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    vertColor = color;
                }
            "
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
                #version 460
                layout(location = 0) in vec4 vertColor;
                layout(location = 0) out vec4 fragColor;

                void main() {
                    fragColor = vertColor;
                }
            "
        }
    }

    let vs = vs::load(device.clone()).expect("Failed to load vertex shader!");
    let fs = fs::load(device.clone()).expect("Failed to load fragment shader!");

    let pipeline = GraphicsPipeline::start()
        .render_pass(PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(swapchain.image_format())],
            ..Default::default()
        })
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .input_assembly_state(InputAssemblyState::new())
        .vertex_shader(
            vs.entry_point("main")
                .expect("Could not find main entry point in vertex shader!"),
            (),
        )
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(
            fs.entry_point("main")
                .expect("Could not find main entry point in fragment shader!"),
            (),
        )
        .build(device.clone())
        .expect("Failed to create graphics pipeline!");

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut attachment_image_views = window_size_dependent_setup(&images, &mut viewport);
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            let dimensions = surface.window().inner_size();
            if dimensions.width == 0 || dimensions.height == 0 {
                return;
            }

            previous_frame_end
                .as_mut()
                .expect("Failed to finish cleanup!")
                .cleanup_finished();

            if recreate_swapchain {
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };

                swapchain = new_swapchain;
                attachment_image_views = window_size_dependent_setup(&new_images, &mut viewport);
                recreate_swapchain = false;
            }

            let (image_num, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .expect("Failed to create command buffer!");

            builder
                .begin_rendering(RenderingInfo {
                    color_attachments: vec![Some(RenderingAttachmentInfo {
                        load_op: LoadOp::Clear,
                        store_op: StoreOp::Store,
                        clear_value: Some([0.0, 0.0, 0.0, 1.0].into()),
                        ..RenderingAttachmentInfo::image_view(
                            attachment_image_views[image_num].clone(),
                        )
                    })],
                    ..Default::default()
                })
                .expect("Failed to begin rendering!")
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .expect("Failed to draw to surface!")
                .end_rendering()
                .expect("Failed to end rendering!");

            let command_buffer = builder.build().expect("Failed to build command buffer!");

            let future = previous_frame_end
                .take()
                .expect("Failed to take value out of previous frame!")
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .expect("Failed to execute command buffer!")
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    eprintln!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    viewport: &mut Viewport,
) -> Vec<Arc<ImageView<SwapchainImage<Window>>>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).expect("Failed to create ImageView!"))
        .collect::<Vec<_>>()
}
