-- Supabase Auth 사용자와 챗봇 대화 저장소 연결
-- 작성자: 김진우

CREATE SCHEMA IF NOT EXISTS private;
REVOKE ALL ON SCHEMA private FROM PUBLIC, anon, authenticated;

ALTER TABLE public.users
    ALTER COLUMN user_id DROP DEFAULT;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'users_auth_user_id_fkey'
          AND conrelid = 'public.users'::regclass
    ) THEN
        ALTER TABLE public.users
            ADD CONSTRAINT users_auth_user_id_fkey
            FOREIGN KEY (user_id)
            REFERENCES auth.users(id)
            ON DELETE CASCADE
            NOT VALID;
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'users_sex_check'
          AND conrelid = 'public.users'::regclass
    ) THEN
        ALTER TABLE public.users
            ADD CONSTRAINT users_sex_check
            CHECK (sex IN ('Male', 'Female'))
            NOT VALID;
    END IF;
END
$$;

ALTER TABLE public.chat_sessions
    ALTER COLUMN user_id DROP DEFAULT,
    ADD COLUMN IF NOT EXISTS title text NOT NULL DEFAULT '새 대화';

ALTER TABLE public.chat_messages
    ADD COLUMN IF NOT EXISTS message_order bigint GENERATED ALWAYS AS IDENTITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chat_messages_role_check'
          AND conrelid = 'public.chat_messages'::regclass
    ) THEN
        ALTER TABLE public.chat_messages
            ADD CONSTRAINT chat_messages_role_check
            CHECK (role IN ('user', 'assistant'));
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS chat_sessions_user_updated_at_idx
    ON public.chat_sessions (user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS chat_messages_session_order_idx
    ON public.chat_messages (session_id, message_order);

CREATE OR REPLACE FUNCTION private.handle_new_auth_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    profile_name text := nullif(btrim(NEW.raw_user_meta_data ->> 'name'), '');
    profile_birth_date text := nullif(NEW.raw_user_meta_data ->> 'birth_date', '');
    profile_sex text := nullif(NEW.raw_user_meta_data ->> 'sex', '');
BEGIN
    -- 관리자 생성처럼 프로필 정보가 없는 Auth 사용자는 가입을 막지 않는다.
    IF profile_name IS NULL OR profile_birth_date IS NULL OR profile_sex IS NULL THEN
        RETURN NEW;
    END IF;

    INSERT INTO public.users (user_id, name, birth_date, sex)
    VALUES (NEW.id, profile_name, profile_birth_date::date, profile_sex)
    ON CONFLICT (user_id) DO NOTHING;

    RETURN NEW;
END;
$$;

REVOKE ALL ON FUNCTION private.handle_new_auth_user() FROM PUBLIC, anon, authenticated;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION private.handle_new_auth_user();

DROP POLICY IF EXISTS users_select_own ON public.users;
CREATE POLICY users_select_own
    ON public.users
    FOR SELECT
    TO authenticated
    USING ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS users_update_own ON public.users;
CREATE POLICY users_update_own
    ON public.users
    FOR UPDATE
    TO authenticated
    USING ((SELECT auth.uid()) = user_id)
    WITH CHECK ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS chat_sessions_select_own ON public.chat_sessions;
CREATE POLICY chat_sessions_select_own
    ON public.chat_sessions
    FOR SELECT
    TO authenticated
    USING ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS chat_sessions_insert_own ON public.chat_sessions;
CREATE POLICY chat_sessions_insert_own
    ON public.chat_sessions
    FOR INSERT
    TO authenticated
    WITH CHECK ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS chat_sessions_update_own ON public.chat_sessions;
CREATE POLICY chat_sessions_update_own
    ON public.chat_sessions
    FOR UPDATE
    TO authenticated
    USING ((SELECT auth.uid()) = user_id)
    WITH CHECK ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS chat_sessions_delete_own ON public.chat_sessions;
CREATE POLICY chat_sessions_delete_own
    ON public.chat_sessions
    FOR DELETE
    TO authenticated
    USING ((SELECT auth.uid()) = user_id);

DROP POLICY IF EXISTS chat_messages_select_own ON public.chat_messages;
CREATE POLICY chat_messages_select_own
    ON public.chat_messages
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1
            FROM public.chat_sessions AS session
            WHERE session.session_id = chat_messages.session_id
              AND session.user_id = (SELECT auth.uid())
        )
    );

DROP POLICY IF EXISTS chat_messages_insert_own ON public.chat_messages;
CREATE POLICY chat_messages_insert_own
    ON public.chat_messages
    FOR INSERT
    TO authenticated
    WITH CHECK (
        EXISTS (
            SELECT 1
            FROM public.chat_sessions AS session
            WHERE session.session_id = chat_messages.session_id
              AND session.user_id = (SELECT auth.uid())
        )
    );

REVOKE ALL ON public.users, public.chat_sessions, public.chat_messages
    FROM anon, authenticated;
GRANT SELECT, UPDATE ON public.users TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.chat_sessions TO authenticated;
GRANT SELECT, INSERT ON public.chat_messages TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE public.chat_messages_message_order_seq
    TO authenticated;

CREATE OR REPLACE FUNCTION public.append_chat_turn(
    p_session_id uuid,
    p_user_content text,
    p_assistant_content text,
    p_summary text
)
RETURNS void
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = ''
AS $$
DECLARE
    updated_count integer;
BEGIN
    UPDATE public.chat_sessions
    SET summary = coalesce(p_summary, ''),
        title = CASE
            WHEN title = '새 대화' THEN left(p_user_content, 60)
            ELSE title
        END,
        updated_at = now()
    WHERE session_id = p_session_id
      AND user_id = (SELECT auth.uid());

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    IF updated_count = 0 THEN
        RAISE EXCEPTION '대화 세션에 접근할 수 없습니다.'
            USING ERRCODE = '42501';
    END IF;

    INSERT INTO public.chat_messages (session_id, role, content)
    VALUES
        (p_session_id, 'user', p_user_content),
        (p_session_id, 'assistant', p_assistant_content);
END;
$$;

REVOKE ALL ON FUNCTION public.append_chat_turn(uuid, text, text, text) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.append_chat_turn(uuid, text, text, text)
    TO authenticated;
