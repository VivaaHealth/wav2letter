import os

recording_ids = [
    "b7cd1798-6496-4815-aeac-86cbae3e8a3f",
    "67ecc208-2219-4ac0-aa87-36c56fbd1a74",
    "706fca8a-6838-4840-85b6-88785c3fe68e",
    "8a9d154c-d4e1-40f4-b5e5-290a4b721db5",
    "cbbe5c6a-21ef-4104-adb4-9abf244c943f",
    "a216caf3-5999-49df-812b-d0cc552cf975",
    "22a90d42-2258-4fe5-a03f-e705da09d0bb",
    "39862230-efe4-4f61-841a-26df4a8be8b8",
    "67ca22eb-a21c-4267-bfd5-215ee0671cde",
    "025ad0a0-d3bf-4acb-bac4-2da373ba3e34",
    "64f0a880-62a3-4b92-a7e6-482f6f7eae87",
    "b5efea7c-7082-4cf7-aea8-0ae22d5108b7",
    "7c809f81-5678-48be-9f70-02eff08a8538",
    "94af5aad-7c84-4455-a1f3-48589ff3d797",
    "78c4cd1f-50d7-4b18-8369-9a0131a9c2b0",
    "cb540cf3-b3b8-403d-92e2-9672a91e8b5a",
    "34d33711-b079-46cd-ae5c-3f728582ee63",
    "df658157-d212-4a93-8de0-fc3dd30aad45",
    "3960c199-71c2-4710-ad62-a0bafdb1b17f",
    "d180f937-b8ce-4016-9b68-dbc5417e6c35",
    "a68d42f5-4b49-44fb-a3c3-04294abd051f",
    "8fe3fe16-96bf-4117-9844-72b7170aac80",
    "2c0430b7-2734-4547-8a93-60eb1acebe2b",
    "d0b8fb04-0f58-4a00-9fda-a0139bcb6eff",
    "5a6e3eff-ed69-4ab7-bc0f-4d2d2c465e7e",
    "b2e29349-2999-413b-a2d9-d431c285ef9d",
    "747352bb-0f87-4b3f-9050-3e8668a7ccbb",
    "7b4168d4-529f-4da8-99b3-473cdcda16d7",
    "c5fc3752-1da6-4095-bb8e-73410f5d0d07",
    "d4c8b850-c9ab-419d-a984-2f4c5510a00b",
    "f2ae6613-3159-4853-864e-0f4e5bbf5562",
    "ca535f92-ce52-4b86-9657-39fb5dbc5513",
    "da297f33-8eb1-47a8-8264-631684e232bf",
    "57df27e2-a906-4718-8cc2-8b8024590e9e",
    "91d4499c-2366-4f6f-ae3a-997b95810169",
    "fb0002bf-72d2-4a5e-955b-32bebc71a11d",
    "64799f6f-9645-4a23-88b2-5bdd35d505e2",
    "a2e0a48d-9386-442e-9f95-243459b599f0",
    "1a6eec9f-277c-4476-8637-c02114b92a33",
    "88e7376c-a71b-404b-bd9b-80ecb8804372",
    "2360b75e-b0ba-4b7e-a3cf-5420e1af907a",
    "c054783e-3e83-42b0-be06-63a542a13d65",
    "26d3e2d2-5a64-4710-8b69-e2477df06e9e",
    "6ed679be-f097-4c24-8123-2f83b9c09c37",
    "c57a212e-0fbb-44e6-8978-ec8e0c898d29",
    "95c4cd5d-770b-41cf-9478-b2f594b69d4c",
    "73ac9df4-1577-44a2-969f-2d347e253753",
    "a3376517-7324-4708-a2c0-3c32b8f20e33",
    "44e9d27b-12d4-4a9c-b843-13596527d172",
    "f4b555b9-ab12-4fa3-8686-e4e24a442632",
    "5887999b-493d-40e1-85e3-e31330cd7781",
    "b9ebcccf-32d3-4e4d-99d0-7bd1f7639a91",
    "ba19111e-dc77-4772-b9cf-6403cfa1ac20",
    "8ae45e74-4de1-4093-924e-38818a665723",
    "2c354d77-99d3-454c-82f1-89d869b90c7a",
    "dc6dca28-15b8-4daf-9f29-82cbfb38fdbb",
    "9807eb95-c4ce-41bb-ae82-88e43436316e",
    "b7b85dc5-dd0a-4068-91bd-1f603d9ca67b",
    "3aa0a8ff-0d58-4eac-86f1-51557391e8e2",
    "338a7baa-755f-4eae-b26f-6a2499447c77",
    "4e47384e-62f1-441c-99d2-4d38da7a051d",
    "48ce9100-b4a9-4b90-9c12-fcc650ae6188",
    "707c48ad-6497-4b23-9a9e-06be42cbcc94",
    "3cf0def7-1813-46ed-a4eb-152f8ac3adf7",
    "8dd17291-af7a-46af-aac4-dfea19861ea1",
    "98f1e5c6-9708-43fb-a86f-98741d94acab",
    "6a5abbe6-fdba-4c16-b8dd-dacf0de5ab5c",
    "650cdf94-a7ad-4753-824e-a7a3b041587d",
    "1d25935b-0a72-4f3c-8ac8-95fadf01827e",
    "a385d13a-e12d-4810-ba67-2151bf2b969c",
    "b28eb260-9edb-4492-a277-8ec7015c90ea",
    "c8384a2d-1498-4ec2-bd6a-42ddc244792d",
    "a270cd3d-df6d-4176-9bd5-4298660d06bf",
    "0b221a3f-deb0-4689-ab65-daee7388b15f",
    "b00b84c4-4f62-4f7a-a3d2-aed5fde381f6",
    "4aa1f108-283b-422e-873b-6756721000b0",
    "a44f9610-117b-45ce-85e9-8a4399d98208",
    "8a88e083-cae3-4771-a05e-e928e475a054",
    "64bdd279-b940-4e81-9845-88a0394755cf",
    "8d5a7b7b-1f43-4e25-b9cc-8c34939d945f",
    "3a889567-a17b-419e-9064-ea5ab84d148c",
    "f31a5339-8dcc-41cf-9e87-5eebd8803efd",
    "e1635e06-aaa4-4aeb-bd9e-1d0514152174",
    "a6ca9c9c-6ec4-4dca-b90d-dfabe0026b91",
    "79c4222f-dd74-4747-82fd-f1f382bb5a72",
    "dc2c9435-a23e-4bd9-af2d-99e0f94c16f8",
    "709db8cb-af99-427d-95d5-81ed7f554d28",
    "78a36760-025a-4bda-811e-efe2d2990545",
    "25b557bf-e8d6-4f3d-be64-d662dd7264be",
    "97b5886a-2dd3-4aa7-ac22-f3a25d05af9a",
    "09ead33d-2f54-49da-ae31-6950fe285ead",
    "3a0320ea-1075-4aeb-aa51-69f8931e9099",
    "723f3617-600e-4597-b5c8-79800c240435",
    "863951c7-a24d-458a-ac5d-7d6fd5fe772a",
    "cfbd03de-064e-4ac2-8c48-78ade1e702fb",
    "322a636d-88d9-4232-9c05-0393db08a8f8",
    "7a944a9e-ee64-4e77-9c7b-16ba4e780bf3",
    "58c36164-9969-4cfb-bbda-4da83f281487",
    "33303c85-6a11-4d63-8360-ea0ca6cec078",
    "cebe6545-15d2-4c26-bd08-f62ddbf37251",
]

local_path = "/home/tetianamyronivska/audio"

for recording_id in recording_ids:
    os.system(
        f"gsutil cp gs://vivaa-staging-files/recordings/{recording_id}.wav {local_path}"
    )
